#!/bin/bash
#SBATCH --job-name=patholia_b2_1_3_HER2
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Patholia — PHASE 2 · BATCH 2.1.3  (HER2 conversion)
set -e

MARKER="HER2"
NEXT_BATCH_NAME="batch2.1.4.sh"

echo "=========================================================="
echo "  BATCH 2.1.3  —  $MARKER conversion"
echo "  Job ID  : $SLURM_JOB_ID | Started : $(date)"
echo "=========================================================="

module load anaconda3/2022.10-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0
source activate "$WORK/.conda_envs/patholia_unistain"
export HF_HOME="$WORK/.hf_cache_patholia"

cd "$WORK/code"
UNISTAIN_CKPT="$WORK/checkpoints/mist_1024/last.ckpt"
OUTPUT_DIR="$WORK/datasets/pannuke_ihc"
mkdir -p "$OUTPUT_DIR/$MARKER"

if [ -f "$OUTPUT_DIR/$MARKER/images.npy" ] && [ -f "$OUTPUT_DIR/$MARKER/labels.npy" ]; then
    echo "[batch2.1.3] $MARKER zaten dönüşmüş — atlanıyor."
else
    python "$WORK/code/veri_boyut/pannuke_to_ihc.py" \
        --output_dir "$OUTPUT_DIR" \
        --checkpoint "$UNISTAIN_CKPT" \
        --markers    "$MARKER" \
        --device     cuda \
        --cache_dir  "$HF_HOME"
    [ -f "$OUTPUT_DIR/$MARKER/images.npy" ] || { echo "[batch2.1.3] HATA: $MARKER üretilmedi"; exit 1; }
    echo "[batch2.1.3] $MARKER OK: $(ls -lh $OUTPUT_DIR/$MARKER/*.npy)"
fi

LOGS_DIR="$WORK/datasets/pannuke_ihc/logs"
mkdir -p "$LOGS_DIR"
cp -a "$HOME/patholia_b2_1_3_${MARKER}_${SLURM_JOB_ID}".{out,err} "$LOGS_DIR/" 2>/dev/null

BATCH_DIR="${BATCH_DIR:-$WORK/code/scripts/hpc/phase2}"
NEXT_BATCH="$BATCH_DIR/$NEXT_BATCH_NAME"
if [ -f "$NEXT_BATCH" ]; then
    NEXT_JID=$(sbatch --parsable --export=ALL,WORK="$WORK",BATCH_DIR="$BATCH_DIR" "$NEXT_BATCH")
    echo "[chain] Submitted $NEXT_BATCH_NAME as job $NEXT_JID"
else
    echo "[chain] HATA: $NEXT_BATCH bulunamadı."
fi
echo "Finished batch 2.1.3 ($MARKER) : $(date)"
