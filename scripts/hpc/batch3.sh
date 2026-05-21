#!/bin/bash
#SBATCH --job-name=patholia_b3
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# =============================================================
#  Patholia — MIST 1024  ·  BATCH 3 / 5   (epochs 40 → 60)
#  Bir önceki: batch2.sh   ·   Sonraki: batch4.sh
# =============================================================

set -e

echo "=========================================================="
echo "  BATCH 3 / 5  —  target max_epochs = 60"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "  Started : $(date)"
echo "  WORK    : $WORK"
echo "=========================================================="

# ── Modules + env ────────────────────────────────────────────
module load anaconda3/2022.10-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0

source activate "$WORK/.conda_envs/patholia_unistain"

export HF_HOME="$WORK/.hf_cache_patholia"
export WANDB_MODE=offline

cd "$WORK/code"

# ── Config ───────────────────────────────────────────────────
CKPT_DIR="$WORK/checkpoints/mist_1024"
mkdir -p "$CKPT_DIR"

MAX_EPOCHS=60

BATCH_DIR="${BATCH_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
NEXT_BATCH="$BATCH_DIR/batch4.sh"

# ── Resume logic ─────────────────────────────────────────────
RESUME_ARG=""
if [ -f "$CKPT_DIR/last.ckpt" ]; then
    RESUME_ARG="--resume_from $CKPT_DIR/last.ckpt"
    echo "[resume] $CKPT_DIR/last.ckpt → Lightning kaldığı epoch'tan devam edecek"
else
    echo "[fresh] last.ckpt yok — beklenmedik durum"
fi

# ── Training ─────────────────────────────────────────────────
python scripts/train/train_mist_1024.py \
    --data_dir   "$WORK/MIST" \
    --stains     HER2 Ki67 ER PR \
    --batch_size 4 \
    --max_epochs $MAX_EPOCHS \
    --ckpt_dir   "$CKPT_DIR" \
    --wandb_name patholia_mist_1024 \
    $RESUME_ARG

# ── Chain: submit next batch ─────────────────────────────────
echo "=========================================================="
echo "  BATCH 3 tamam — batch4.sh kuyruğa atılıyor..."
echo "=========================================================="

if [ -f "$NEXT_BATCH" ]; then
    NEXT_JID=$(sbatch --parsable --export=ALL,WORK="$WORK",BATCH_DIR="$BATCH_DIR" "$NEXT_BATCH")
    echo "[chain] Submitted batch4.sh as job $NEXT_JID"
else
    echo "[chain] HATA: $NEXT_BATCH bulunamadı — zincir kırıldı."
fi

echo "Finished batch 3 : $(date)"
