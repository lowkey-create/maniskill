#!/usr/bin/env bash
set -euo pipefail

output=${1:?Specify a corpus directory}
batch=${2:-144}
pilot=${3:-0}
python=${PYTHON:-python}
mkdir -p "$output"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
pids=()
for worker in 0 1; do
    "$python" -m spatial_motion.generate_corpus --pattern expanded-v2 \
        --output "$output/worker_$worker" --num-envs "$batch" --pilot "$pilot" \
        --video-envs 1 --rgb-writer-threads 4 --worker-count 2 --worker-index "$worker" \
        >"$output/worker_$worker.log" 2>&1 &
    pids+=("$!")
done
trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
(( failed == 0 ))

"$python" - "$output" "$pilot" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
pilot = int(sys.argv[2])
expected = {"train": 7728, "validation": 1008, "test": 1344}
if pilot:
    expected = {"train": 18 * pilot, "validation": 18 * pilot, "test": 22 * pilot}
entries = []
for worker in sorted(root.glob("worker_*/manifest.json")):
    for entry in json.loads(worker.read_text()):
        entry["shard"] = f"{worker.parent.name}/{entry['shard']}"
        entries.append(entry)
counts = Counter(entry["split"] for entry in entries)
if counts != expected or len({entry["id"] for entry in entries}) != len(entries):
    raise RuntimeError(f"Incomplete corpus: {dict(counts)}")
temporary = root / "manifest.tmp"
temporary.write_text(json.dumps(entries, indent=2) + "\n")
temporary.replace(root / "manifest.json")
print(f"Verified corpus: {dict(counts)}")
PY
"$python" -m spatial_motion.plot_corpus "$output" --output "$output/trajectories.png"
