"""Optional Mesa EGL rendering of the live ManiSkill Panda on CPU."""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

VERTEX = """
#version 330
uniform mat4 model;
uniform mat4 view_projection;
in vec3 point;
in vec3 normal;
out vec3 world_normal;
out vec3 world_position;
void main() {
    vec4 world = model * vec4(point, 1.0);
    world_normal = mat3(model) * normal;
    world_position = world.xyz;
    gl_Position = view_projection * world;
}
"""
FRAGMENT = """
#version 330
uniform vec3 color;
uniform bool floor_surface;
in vec3 world_normal;
in vec3 world_position;
out vec4 pixel;
void main() {
    vec3 n = normalize(world_normal);
    float light = 0.50 + 0.50 * max(dot(n, normalize(vec3(0.4, -0.6, 1.0))), 0.0);
    vec3 base = color;
    if (floor_surface) {
        vec2 grid = abs(fract(world_position.xy / 0.1 + 0.5) - 0.5);
        if (max(grid.x, grid.y) > 0.48) base *= 0.85;
    }
    pixel = vec4(base * light, 1.0);
}
"""
# Columns express OpenCV camera axes in the SAPIEN camera frame.
CV_TO_SAPIEN = np.array([[0, 0, 1, 0], [-1, 0, 0, 0],
                        [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
CV_TO_GL = np.diag([1, -1, -1, 1])


class SoftwareRenderer:
    def __init__(self, env):
        # Keep optional packages local to this project rather than editing .venv.
        dependencies = Path(__file__).with_name(".render_deps")
        if dependencies.is_dir():
            sys.path.insert(0, str(dependencies))
        import moderngl

        self.context = moderngl.create_standalone_context(backend="egl")
        try:
            self.context.enable(moderngl.DEPTH_TEST)
            self.program = self.context.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
            self.cameras = env._default_sensor_configs
            self.buffers = {cam.uid: self.context.simple_framebuffer((cam.width, cam.height))
                            for cam in self.cameras}
            self.meshes = []
            robot = env.agent.robot
            urdf = Path(env.agent.urdf_path)
            for element in ET.parse(urdf).getroot().findall("link"):
                link = robot.links_map[element.attrib["name"]]
                for visual in element.findall("visual"):
                    mesh = visual.find("geometry/mesh")
                    if mesh is None:
                        raise ValueError(f"Unsupported Panda visual geometry in {link.name}")
                    origin = visual.find("origin")
                    transform = np.eye(4)
                    if origin is not None:
                        transform[:3, :3] = Rotation.from_euler(
                            "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                        ).as_matrix()
                        transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                    scene = trimesh.load_scene(urdf.parent / mesh.attrib["filename"])
                    scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                    for part in scene.dump():
                        part.apply_scale(scale)
                        material = getattr(part.visual, "material", None)
                        color = getattr(material, "main_color", [205, 208, 214, 255])
                        self._add_mesh(part, link, transform, np.asarray(color[:3]) / 255)
            floor = trimesh.creation.box(extents=[4, 4, 0.01])
            floor.apply_translation([0, 0, -0.005])
            self._add_mesh(floor, None, np.eye(4), (0.50, 0.52, 0.56))
            self.device = self.context.info["GL_RENDERER"]
        except BaseException:
            self.close()
            raise

    def _add_mesh(self, mesh, link, transform, color):
        vertices = np.column_stack((mesh.vertices, mesh.vertex_normals)).astype("f4")
        vbo = self.context.buffer(vertices.tobytes())
        ibo = self.context.buffer(mesh.faces.astype("i4").tobytes())
        vao = self.context.vertex_array(self.program, [(vbo, "3f 3f", "point", "normal")], ibo)
        self.meshes.append((link, transform, tuple(color), vao, vbo, ibo))

    def capture(self):
        data, params = {}, {}
        robot_models = [
            origin if link is None else link.pose.to_transformation_matrix()[0].cpu().numpy() @ origin
            for link, origin, *_ in self.meshes
        ]
        for cam in self.cameras:
            pose = cam.pose if cam.mount is None else cam.mount.pose * cam.pose
            model = pose.to_transformation_matrix()[0].cpu().numpy()
            world_to_cv = np.linalg.inv(model @ CV_TO_SAPIEN)
            f = 1 / np.tan(cam.fov / 2)
            aspect = cam.width / cam.height
            near, far = cam.near, cam.far
            projection = np.array([
                [f / aspect, 0, 0, 0], [0, f, 0, 0],
                [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
                [0, 0, -1, 0],
            ])
            buffer = self.buffers[cam.uid]
            buffer.use()
            buffer.clear(0.78, 0.83, 0.89, depth=1)
            self.program["view_projection"].write((projection @ CV_TO_GL @ world_to_cv).T.astype("f4").tobytes())
            for geometry, (link, _, color, vao, _, _) in zip(robot_models, self.meshes):
                self.program["model"].write(geometry.T.astype("f4").tobytes())
                self.program["color"].value = color
                self.program["floor_surface"].value = link is None
                vao.render()
            pixels = np.frombuffer(buffer.read(components=3), dtype=np.uint8)
            data[cam.uid] = {"rgb": pixels.reshape(1, cam.height, cam.width, 3)[:, ::-1].copy()}
            focal = f * cam.height / 2
            params[cam.uid] = {
                "intrinsic_cv": np.array([[[focal, 0, cam.width / 2],
                                           [0, focal, cam.height / 2], [0, 0, 1]]]),
                "extrinsic_cv": world_to_cv[None, :3],
                "cam2world_gl": (model @ CV_TO_SAPIEN @ CV_TO_GL)[None],
            }
        return {"sensor_data": data, "sensor_param": params}

    def close(self):
        self.context.release()
