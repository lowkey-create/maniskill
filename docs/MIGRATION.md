# SpatialMotion server migration

The release `spatialmotion-corpus-240-v1` contains the data and artifacts for the absolute-pose version of SpatialMotion, including 320 accepted demonstrations and their complete two-camera RGB recordings.

Use Git to fetch the code and documentation. Download the four archives and `SHA256SUMS` from the [GitHub release](https://github.com/lowkey-create/maniskill/releases/tag/spatialmotion-corpus-240-v1) into a directory named `migration-assets` at the repository root. With GitHub CLI, the download command is:

```bash
git clone https://github.com/lowkey-create/maniskill.git
cd maniskill
git switch --detach spatialmotion-corpus-240-v1
gh release download spatialmotion-corpus-240-v1 --dir migration-assets
(cd migration-assets && sha256sum -c SHA256SUMS)
for archive in migration-assets/*.tar.gz; do
  tar -xzf "$archive" -C .
done
```

The restored corpus is at `spatial_motion/run_absolute_5090/corpus_240/`. Its `manifest.json` maps episode IDs and instructions to each HDF5 shard and slot. The extras archive contains the current CPU and GPU reference runs, the source HDF5 inspection and playback helpers, and the successful DrawSVG flower reference run.

The Python environment is provisioned separately on the target host. See [SpatialMotion setup](../spatial_motion/README.md) and [the corpus field reference](SpatialMotion_采集数据说明.docx).
