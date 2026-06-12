#!/bin/bash
#SBATCH --job-name=patholia_b2_4
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Patholia — PHASE 2 · BATCH 2.4  (HoVerNext train chunk 3 / 7)
# Hedef: ~60K → 90K
set -e

TARGET_STEPS=90000
NEXT_BATCH_NAME="batch2.5.sh"

echo "=========================================================="
echo "  BATCH 2.4  —  HoVerNext train chunk 3/7 (→ ${TARGET_STEPS} step)"
echo "  Job ID  : $SLURM_JOB_ID | Started: $(date)"
echo "=========================================================="

module load anaconda3/2022.10-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0
source activate "$WORK/.conda_envs/patholia_hovernext"
export HF_HOME="$WORK/.hf_cache_patholia"
export WANDB_MODE=offline

HOVERNEXT_DIR="$WORK/hover_next_ihc/hover_next_train-main"
DATA_DIR="$WORK/datasets/pannuke_ihc"
EXP_NAME="ihc_pannuke_convnextv2_tiny"
EXP_DIR="$WORK/checkpoints/$EXP_NAME"

ls "$EXP_DIR"/*.pth 1>/dev/null 2>&1 || { echo "[batch2.4] HATA: önceki ckpt yok"; exit 1; }
LAST_CKPT=$(ls -t "$EXP_DIR"/*.pth | head -1)
echo "[batch2.4] Resume: $LAST_CKPT"

CONFIG_FILE="$WORK/checkpoints/hovernext_config_b2_4.toml"
cp "$HOVERNEXT_DIR/sample_configs/train_ihc_pannuke.toml" "$CONFIG_FILE"
sed -i "s|^data_path.*|data_path = \"$DATA_DIR\"|" "$CONFIG_FILE"
sed -i "s|^training_steps.*|training_steps = $TARGET_STEPS|" "$CONFIG_FILE"
sed -i "s|^experiment.*|experiment = \"$EXP_NAME\"|" "$CONFIG_FILE"
sed -i "s|^checkpoint_path.*|checkpoint_path = \"$LAST_CKPT\"|" "$CONFIG_FILE"

cd "$HOVERNEXT_DIR"
torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py --config "$CONFIG_FILE"

ls -lh "$EXP_DIR"/*.pth 2>/dev/null | tail -3 || echo "  (yok)"

LOGS_DIR="$WORK/datasets/pannuke_ihc/logs"
mkdir -p "$LOGS_DIR"
cp -a "$HOME/patholia_b2_4_${SLURM_JOB_ID}".{out,err} "$LOGS_DIR/" 2>/dev/null

BATCH_DIR="${BATCH_DIR:-$WORK/code/scripts/hpc/phase2}"
NEXT_BATCH="$BATCH_DIR/$NEXT_BATCH_NAME"
if [ -f "$NEXT_BATCH" ]; then
    NEXT_JID=$(sbatch --parsable --export=ALL,WORK="$WORK",BATCH_DIR="$BATCH_DIR" "$NEXT_BATCH")
    echo "[chain] Submitted $NEXT_BATCH_NAME as job $NEXT_JID"
fi
echo "Finished batch 2.4 : $(date)"
