#!/bin/bash
#SBATCH --job-name=patholia_b2_2
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Patholia — PHASE 2 · BATCH 2.2  (HoVerNext train chunk 1 / 7)
# Hedef step: 0 → 30K  (toplam 200K)
set -e

TARGET_STEPS=30000
NEXT_BATCH_NAME="batch2.3.sh"

echo "=========================================================="
echo "  BATCH 2.2  —  HoVerNext train chunk 1/7 (0 → ${TARGET_STEPS} step)"
echo "  Job ID  : $SLURM_JOB_ID | Node: $(hostname) | Started: $(date)"
echo "  WORK    : $WORK"
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
mkdir -p "$EXP_DIR"

# Veri doğrula
for M in ER PR HER2 Ki67; do
    [ -f "$DATA_DIR/$M/images.npy" ] || { echo "[batch2.2] HATA: $DATA_DIR/$M/images.npy yok"; exit 1; }
done

# Başlangıç checkpoint'i seç:
#   1) Önceki batch2.2 run'ı yarıda kesilmişse: $EXP_DIR'daki son .pth (resume)
#   2) İlk çalıştırma: setup2.sh'ın indirdiği PanNuke pretrained (fine-tune)
#   3) İkisi de yoksa → DUR (sıfırdan eğitime düşülmüyor)
PRETRAINED_CKPT="$WORK/pretrained_hovernext/pannuke_convnextv2_tiny_3/best_model"
LAST_CKPT=""

if ls "$EXP_DIR"/*.pth 1>/dev/null 2>&1; then
    LAST_CKPT=$(ls -t "$EXP_DIR"/*.pth | head -1)
    echo "[batch2.2] Resume: $LAST_CKPT (mevcut çalışmadan)"
elif [ -f "$PRETRAINED_CKPT" ]; then
    LAST_CKPT="$PRETRAINED_CKPT"
    echo "[batch2.2] Fine-tune: $PRETRAINED_CKPT (PanNuke pretrained)"
else
    echo "[batch2.2] HATA: pretrained checkpoint bulunamadı!"
    echo "          Beklenen: $PRETRAINED_CKPT"
    echo "          setup2.sh tamamlandı mı? Drive klasöründe"
    echo "          'pannuke_convnextv2_tiny_3.zip' var mı?"
    echo "          Sıfırdan eğitime DÜŞÜLMÜYOR — zincir burada duruyor."
    exit 1
fi

# Config TOML
CONFIG_FILE="$WORK/checkpoints/hovernext_config_b2_2.toml"
cp "$HOVERNEXT_DIR/sample_configs/train_ihc_pannuke.toml" "$CONFIG_FILE"
sed -i "s|^data_path.*|data_path = \"$DATA_DIR\"|" "$CONFIG_FILE"
sed -i "s|^training_steps.*|training_steps = $TARGET_STEPS|" "$CONFIG_FILE"
sed -i "s|^experiment.*|experiment = \"$EXP_NAME\"|" "$CONFIG_FILE"
[ -n "$LAST_CKPT" ] && sed -i "s|^checkpoint_path.*|checkpoint_path = \"$LAST_CKPT\"|" "$CONFIG_FILE"

echo "[batch2.2] training_steps = $TARGET_STEPS  |  data_path = $DATA_DIR"

cd "$HOVERNEXT_DIR"
torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py --config "$CONFIG_FILE"

echo "[batch2.2] Checkpoint'ler:"
ls -lh "$EXP_DIR"/*.pth 2>/dev/null | tail -3 || echo "  (yok)"

# Loglarını container'a kopyala
LOGS_DIR="$WORK/datasets/pannuke_ihc/logs"
mkdir -p "$LOGS_DIR"
cp -a "$HOME/patholia_b2_2_${SLURM_JOB_ID}".{out,err} "$LOGS_DIR/" 2>/dev/null

# Chain
BATCH_DIR="${BATCH_DIR:-$WORK/code/scripts/hpc/phase2}"
NEXT_BATCH="$BATCH_DIR/$NEXT_BATCH_NAME"
if [ -f "$NEXT_BATCH" ]; then
    NEXT_JID=$(sbatch --parsable --export=ALL,WORK="$WORK",BATCH_DIR="$BATCH_DIR" "$NEXT_BATCH")
    echo "[chain] Submitted $NEXT_BATCH_NAME as job $NEXT_JID"
else
    echo "[chain] HATA: $NEXT_BATCH bulunamadı."
fi
echo "Finished batch 2.2 : $(date)"
