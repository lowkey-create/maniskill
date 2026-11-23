
import gymnasium as gym
import mani_skill.envs
import imageio

# 自定义心形 SVG 路径
svg_path = (
    "M 50 80 "
    "C 5 48 20 15 50 38 "
    "C 80 15 95 48 50 80 Z"
)

env = gym.make(
    "DrawSVG-v1",
    svg=svg_path,
    obs_mode="rgb",
    render_mode="rgb_array",
    sim_backend="physx_cpu",
)

obs, info = env.reset(seed=42)

# 官方环境生成的目标轨迹
print("目标轨迹点：")
print(env.unwrapped.points[0].cpu().numpy())

# 保存目标图形的仿真画面
image = env.render()
if hasattr(image, "cpu"):
    image = image.cpu().numpy()
if image.ndim == 4:
    image = image[0]

imageio.imwrite("svg_goal.png", image)

env.close()
print("目标图形已保存到 svg_goal.png")
