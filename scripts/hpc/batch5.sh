#!/bin/bash
#SBATCH --job-name=patholia_b5
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# =============================================================
#  Patholia — MIST 1024  ·  BATCH 5 / 5   (epochs 80 → 100)
#  Bir önceki: batch4.sh
#  SONRAKİ YOK — eğitim burada biter.
# =============================================================

set -e

echo "=========================================================="
echo "  BATCH 5 / 5  —  target max_epochs = 100  (FINAL)"
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

MAX_EPOCHS=100

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

# ── End of chain ─────────────────────────────────────────────
echo "=========================================================="
echo "  BATCH 5 (FINAL) tamam — toplam 100 epoch eğitim BİTTİ."
echo "  Checkpoint'ler: $CKPT_DIR"
echo "    - last.ckpt                     (epoch 99 state)"
echo "    - mist_1024_epoch=NNN_*.ckpt    (en iyi val/lpips skorlu top-3)"
echo "=========================================================="

ls -lh "$CKPT_DIR"

echo "Finished batch 5 (FINAL) : $(date)"
