#!/bin/bash
# One full P3 simulation run on a Turing GPU node, plus the FPV video.
#
#   cd ~/scratch/rbe595/p3 && sbatch run_sim.sbatch                 # defaults: a3_train, policy_template:WindowRacer
#   sbatch run_sim.sbatch --scene /path/to/a3_test --max-s 90       # anything after the file name goes to Wrapper.py
#
# Submit from the repo root (the folder with Code/, scene/, p3_contract.json).
# Everything for one job lands in runs/<job id>/:
#   run.json, run.png, run_log.txt (score + every mode change), fpv/frame_*.png, FPVVideo.mp4
#
#SBATCH -J p3_sim
#SBATCH -p academic
#SBATCH --account rbe595
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o slurm-p3_sim-%j.out

set -eo pipefail

# --- environment -------------------------------------------------------------
module load cuda12.6/toolkit/12.6.3          # matches torch 2.6.0+cu126; gsplat compiles with it
PY=${PY:-$HOME/.conda/envs/aerial_robotics/bin/python}
export PATH="$(dirname "$PY"):$PATH"          # the env's python, ninja and friends, no 'conda activate' needed
cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "node: $(hostname)   job: ${SLURM_JOB_ID:-local}   repo: $PWD"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PY" - <<'EOF'
import torch, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else "no GPU visible to torch")
EOF

# --- the run -------------------------------------------------------------------
OUT="runs/${SLURM_JOB_ID:-local_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
START=$(date +%s)
set +e
"$PY" Code/Wrapper.py --pack . --worker-python "$PY" --log "$OUT/run.json" "$@" 2>&1 | tee "$OUT/run_log.txt"
STATUS=${PIPESTATUS[0]}
set -e
echo "Wrapper.py exit status $STATUS, wall clock $(( $(date +%s) - START )) s"

# Wrapper.py writes run.png next to run.json; the FPV recorder writes ./fpv/<timestamp>/
FPV=$(ls -td fpv/*/ 2>/dev/null | head -1 || true)
if [ -z "$FPV" ]; then
    echo "no FPV frames found"; exit "$STATUS"
fi
mv "$FPV" "$OUT/fpv"
echo "frames: $(ls "$OUT/fpv" | wc -l) in $OUT/fpv"

# --- the FPV video (RGB | depth | detections, side by side) ------------------
module load ffmpeg >/dev/null 2>&1 || true
if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -loglevel error -y -framerate 15 -i "$OUT/fpv/frame_%05d.png" \
        -vf scale=iw*2:ih*2:flags=neighbor -pix_fmt yuv420p "$OUT/FPVVideo.mp4"
else
    # no ffmpeg on the node: OpenCV writes the same video (frames doubled in size, 15 fps)
    "$PY" - "$OUT" <<'EOF'
import glob, sys, cv2
out = sys.argv[1]
frames = sorted(glob.glob(out + "/fpv/frame_*.png"))
h, w = cv2.imread(frames[0]).shape[:2]
vw = cv2.VideoWriter(out + "/FPVVideo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (2 * w, 2 * h))
for f in frames:
    vw.write(cv2.resize(cv2.imread(f), (2 * w, 2 * h), interpolation=cv2.INTER_NEAREST))
vw.release()
EOF
fi
echo "video: $OUT/FPVVideo.mp4"
grep -E "outcome|gates passed|collision|closest approach|your policy|rendering" "$OUT/run_log.txt" || true
exit "$STATUS"
