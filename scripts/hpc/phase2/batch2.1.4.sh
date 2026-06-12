#!/bin/bash
#SBATCH --job-name=patholia_b2_1_4_Ki67
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Patholia — PHASE 2 · BATCH 2.1.4  (Ki67 conversion, son conversion)
# Bittikten sonra batch2.2 (HoVerNext train chunk 1) tetiklenir.
set -e

MARKER="Ki67"
NEXT_BATCH_NAME="batch2.2.sh"

echo "=========================================================="
echo "  BATCH 2.1.4  —  $MARKER conversion (son conversion batch)"
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
    echo "[batch2.1.4] $MARKER zaten dönüşmüş — atlanıyor."
else
    python "$WORK/code/veri_boyut/pannuke_to_ihc.py" \
        --output_dir "$OUTPUT_DIR" \
        --checkpoint "$UNISTAIN_CKPT" \
        --markers    "$MARKER" \
        --device     cuda \
        --cache_dir  "$HF_HOME"
    [ -f "$OUTPUT_DIR/$MARKER/images.npy" ] || { echo "[batch2.1.4] HATA: $MARKER üretilmedi"; exit 1; }
fi

# 4 marker'ın hepsinin var olduğunu doğrula
echo "[batch2.1.4] Tüm marker'lar doğrulanıyor..."
MISSING=0
for M in ER PR HER2 Ki67; do
    if [ -f "$OUTPUT_DIR/$M/images.npy" ]; then
        size=$(du -h "$OUTPUT_DIR/$M/images.npy" | cut -f1)
        printf "  %-5s : OK (%s)\n" "$M" "$size"
    else
        echo "  $M : MISSING"
        MISSING=$((MISSING + 1))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo "[batch2.1.4] HATA: $MISSING marker eksik — eğitim başlatılamaz."
    exit 1
fi

# Loglarını container'a kopyala (süreç boyunca biriksin)
LOGS_DIR="$WORK/datasets/pannuke_ihc/logs"
mkdir -p "$LOGS_DIR"
cp -a "$HOME/patholia_b2_1_4_${MARKER}_${SLURM_JOB_ID}".{out,err} "$LOGS_DIR/" 2>/dev/null
# Phase 1 ve setup loglarını da bu noktada container'a getir
cp -a /scratch/prj/hpc_training/patholia/logs/* "$LOGS_DIR/" 2>/dev/null
cp -a $HOME/patholia_b1_*.out $HOME/patholia_b1_*.err "$LOGS_DIR/" 2>/dev/null
cp -a $HOME/patholia_b2_*.out $HOME/patholia_b2_*.err "$LOGS_DIR/" 2>/dev/null
cp -a $HOME/patholia_b3_*.out $HOME/patholia_b3_*.err "$LOGS_DIR/" 2>/dev/null
cp -a $HOME/patholia_b4_*.out $HOME/patholia_b4_*.err "$LOGS_DIR/" 2>/dev/null
cp -a $HOME/patholia_b5_*.out $HOME/patholia_b5_*.err "$LOGS_DIR/" 2>/dev/null

# Chain → batch2.2 (HoVerNext training start)
BATCH_DIR="${BATCH_DIR:-$WORK/code/scripts/hpc/phase2}"
NEXT_BATCH="$BATCH_DIR/$NEXT_BATCH_NAME"

echo "=========================================================="
echo "  Tüm 4 marker dönüşüm tamam — HoVerNext eğitim zinciri başlıyor..."
echo "=========================================================="

if [ -f "$NEXT_BATCH" ]; then
    NEXT_JID=$(sbatch --parsable --export=ALL,WORK="$WORK",BATCH_DIR="$BATCH_DIR" "$NEXT_BATCH")
    echo "[chain] Submitted $NEXT_BATCH_NAME as job $NEXT_JID"
else
    echo "[chain] HATA: $NEXT_BATCH bulunamadı."
fi
echo "Finished batch 2.1.4 ($MARKER) : $(date)"
