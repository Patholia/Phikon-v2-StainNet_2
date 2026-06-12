#!/bin/bash
#SBATCH --job-name=patholia_setup2
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prj/hpc_training/patholia/logs/setup2_%j.out
#SBATCH --error=/scratch/prj/hpc_training/patholia/logs/setup2_%j.err

# =============================================================
#  Patholia — PHASE 2 SETUP
#
#  Faz 1 (UNIStainNet 100 epoch) bitti, last.ckpt hazır.
#  Bu setup faz 2'yi kurar:
#    1. hover_next_ihc repo'sunu klonlar
#    2. Yeni conda env (patholia_hovernext) + bağımlılıklar
#    3. Conversion için gerekli ekstra paketleri (datasets, cv2) faz 1 env'ine ekler
#    4. batch2.1.sh'i kuyruğa atar → veri dönüşümü zinciri başlar
#
#  KCL'de:
#      curl -O https://raw.githubusercontent.com/Patholia/Phikon-v2-StainNet_2/main/scripts/hpc/phase2/setup2.sh
#      sbatch setup2.sh
# =============================================================

set -e

echo "=========================================================="
echo "  PATHOLIA PHASE 2 SETUP — $(date)"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "=========================================================="

# ── Modules ──────────────────────────────────────────────────
module load anaconda3/2022.10-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0

mkdir -p /scratch/prj/hpc_training/patholia/logs

# ── 0. Work directory (faz 1 ile aynı) ───────────────────────
export WORK=/scratch/prj/hpc_training/patholia/patholia_unistain
echo "[setup2] WORK = $WORK"
cd "$WORK"

# ── 1. Faz 1 checkpoint'ini doğrula ──────────────────────────
UNISTAIN_CKPT="$WORK/checkpoints/mist_1024/last.ckpt"
if [ ! -f "$UNISTAIN_CKPT" ]; then
    echo "[setup2] HATA: UNIStainNet checkpoint bulunamadı!"
    echo "         Beklenen: $UNISTAIN_CKPT"
    echo "         Faz 1 (batch1-5) tamamlanmamış olabilir."
    exit 1
fi
echo "[setup2] UNIStainNet checkpoint OK: $UNISTAIN_CKPT ($(du -h $UNISTAIN_CKPT | cut -f1))"

# ── 2. UNIStainNet repo'yu güncelle (pannuke_to_ihc.py için) ──
cd "$WORK/code"
echo "[setup2] UNIStainNet repo güncelleniyor..."
git pull --ff-only || echo "[setup2] git pull failed (continuing)"

# pannuke_to_ihc.py varlığını doğrula
PANNUKE_SCRIPT="$WORK/code/veri_boyut/pannuke_to_ihc.py"
if [ ! -f "$PANNUKE_SCRIPT" ]; then
    echo "[setup2] HATA: $PANNUKE_SCRIPT bulunamadı."
    echo "         GitHub'a push edildi mi?"
    exit 1
fi
echo "[setup2] Conversion script OK: $PANNUKE_SCRIPT"

# ── 3. hover_next_ihc repo'yu klonla ─────────────────────────
HOVERNEXT_DIR="$WORK/hover_next_ihc"
if [ ! -d "$HOVERNEXT_DIR/.git" ]; then
    echo "[setup2] hover_next_ihc klonlanıyor..."
    git clone https://github.com/Patholia/hover_next_ihc.git "$HOVERNEXT_DIR"
else
    echo "[setup2] hover_next_ihc zaten var — pull yapılıyor..."
    (cd "$HOVERNEXT_DIR" && git pull --ff-only) || echo "[setup2] pull failed (continuing)"
fi

# Klasör yapısını doğrula
if [ ! -d "$HOVERNEXT_DIR/hover_next_train-main" ]; then
    echo "[setup2] HATA: hover_next_train-main klasörü bulunamadı."
    echo "         Repo yapısını kontrol et: $HOVERNEXT_DIR"
    exit 1
fi
echo "[setup2] hover_next_train-main OK"

# ── 4. Faz 1 env'ine ekstra paketler (conversion için) ────────
source activate "$WORK/.conda_envs/patholia_unistain"
echo "[setup2] Conversion paketleri (datasets, opencv) ekleniyor..."
pip install --quiet datasets opencv-python-headless

# Verify
python -c "from datasets import load_dataset; import cv2; print('[setup2] conversion env OK')"

# ── 5. HoVerNext için yeni conda env ─────────────────────────
HOVERNEXT_ENV="$WORK/.conda_envs/patholia_hovernext"
if [ ! -d "$HOVERNEXT_ENV" ]; then
    echo "[setup2] HoVerNext conda env yaratılıyor..."
    conda create -p "$HOVERNEXT_ENV" python=3.10 -y
else
    echo "[setup2] HoVerNext env zaten var — paket güncellemesi yapılacak."
fi

conda deactivate
source activate "$HOVERNEXT_ENV"

# CUDA sürümüne göre pytorch
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
echo "[setup2] CUDA $CUDA_VER tespit edildi"

pip install --upgrade pip
if [ "$MAJOR" -ge 12 ]; then
    pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
fi

# hover_next_train requirements — ağır paketler
echo "[setup2] HoVerNext bağımlılıkları kuruluyor (10-15 dk)..."
pip install --no-cache-dir -r "$HOVERNEXT_DIR/hover_next_train-main/requirements.txt"

# Verify
python -c "
import torch
import timm
import segmentation_models_pytorch
import mahotas
import toml
print('[setup2] HoVerNext env OK | CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))
"

# ── 6. HuggingFace cache (faz 1 ile aynı) ────────────────────
export HF_HOME="$WORK/.hf_cache_patholia"
mkdir -p "$HF_HOME"

# PanNuke (HF'den) önbelleğe almak — conversion script aynı cache'i kullanacak
echo "[setup2] PanNuke HF cache hazırlanıyor (büyük dataset, 10-20 dk)..."
conda deactivate
source activate "$WORK/.conda_envs/patholia_unistain"
export HF_HOME="$WORK/.hf_cache_patholia"

python -c "
from datasets import load_dataset
print('[setup2] PanNuke indiriliyor...')
ds = load_dataset('RationAI/PanNuke', cache_dir='$HF_HOME')
print('[setup2] PanNuke cached:', list(ds.keys()))
"

# ── 7. PanNuke pretrained HoVerNext checkpoint'i indir ───────
#  HoVerNext-IHC fine-tuning için başlangıç noktası.
#  Öncelik 1: Zenodo (resmi kaynak, ~133 MB)
#  Öncelik 2 (fallback): Drive (Zenodo erişilemezse)
PRETRAINED_DIR="$WORK/pretrained_hovernext"
PRETRAINED_CKPT="$PRETRAINED_DIR/pannuke_convnextv2_tiny_3/best_model"
mkdir -p "$PRETRAINED_DIR"

if [ ! -f "$PRETRAINED_CKPT" ]; then
    cd "$PRETRAINED_DIR"
    ZIP_OK=0

    # ── 7a. Önce Zenodo'dan dene ──
    echo "[setup2] Zenodo'dan pannuke_convnextv2_tiny_3 indiriliyor..."
    if curl -fsSL --max-time 600 \
        -o pannuke_convnextv2_tiny_3.zip \
        "https://zenodo.org/records/10635618/files/pannuke_convnextv2_tiny_3.zip?download=1"; then
        if [ -s pannuke_convnextv2_tiny_3.zip ]; then
            echo "[setup2] Zenodo OK."
            ZIP_OK=1
        else
            echo "[setup2] Zenodo: indirme boş geldi."
        fi
    else
        echo "[setup2] Zenodo erişilemedi."
    fi

    # ── 7b. Zenodo başarısızsa Drive'dan dene ──
    if [ "$ZIP_OK" -eq 0 ]; then
        echo "[setup2] Fallback: Drive'dan indiriliyor..."
        rm -f pannuke_convnextv2_tiny_3.zip
        # Drive klasörü: 1vu4FXY7lbQoaeOo78kWkwL5mEZb5onil
        gdown --folder \
            https://drive.google.com/drive/folders/1vu4FXY7lbQoaeOo78kWkwL5mEZb5onil \
            -O "$PRETRAINED_DIR/drive_dl"
        # gdown --folder klasör adıyla iniyor, .zip'i çıkart
        FOUND=$(find "$PRETRAINED_DIR/drive_dl" -name "pannuke_convnextv2_tiny_3.zip" | head -1)
        if [ -n "$FOUND" ]; then
            mv "$FOUND" "$PRETRAINED_DIR/pannuke_convnextv2_tiny_3.zip"
            rm -rf "$PRETRAINED_DIR/drive_dl"
            ZIP_OK=1
            echo "[setup2] Drive OK."
        else
            echo "[setup2] HATA: Drive'da da pannuke_convnextv2_tiny_3.zip bulunamadı."
            exit 1
        fi
    fi

    # ── 7c. Zip'i aç ──
    if [ "$ZIP_OK" -eq 1 ]; then
        cd "$PRETRAINED_DIR"
        unzip -q pannuke_convnextv2_tiny_3.zip
        rm pannuke_convnextv2_tiny_3.zip
    fi

    if [ ! -f "$PRETRAINED_CKPT" ]; then
        echo "[setup2] HATA: $PRETRAINED_CKPT beklenen yerde değil. Klasör içeriği:"
        find "$PRETRAINED_DIR" -maxdepth 3 | head -20
        echo
        echo "[setup2] Beklenen yapı: pretrained_hovernext/pannuke_convnextv2_tiny_3/best_model"
        echo "[setup2] Zip içeriği farklı path'te açılmış olabilir — manuel olarak"
        echo "         '$PRETRAINED_CKPT' yoluna 'best_model' dosyasını taşı."
        exit 1
    fi
fi

echo "[setup2] Pretrained checkpoint OK: $PRETRAINED_CKPT"
echo "         (HoVerNext-IHC bu noktadan fine-tune edilecek)"

# ── 8. Batch dosyalarını çalıştırılabilir yap ────────────────
chmod +x "$WORK/code/scripts/hpc/phase2/"batch2.*.sh
echo "[setup2] Phase 2 batch dosyaları hazır: $(ls $WORK/code/scripts/hpc/phase2/batch2.*.sh | wc -l) dosya"

# ── 8. batch2.1 (conversion) kuyruğa at ──────────────────────
echo ""
echo "=========================================================="
echo "  SETUP2 COMPLETE — faz 2 zinciri başlatılıyor"
echo "=========================================================="
echo ""

BATCH211_JID=$(sbatch --parsable \
    --export=ALL,WORK="$WORK",BATCH_DIR="$WORK/code/scripts/hpc/phase2" \
    "$WORK/code/scripts/hpc/phase2/batch2.1.1.sh")

echo "[setup2] batch2.1.1.sh submitted as job $BATCH211_JID"
echo ""
echo "Zincir (11 batch — toplam ~7-12 gün):"
echo "  batch2.1.1  (ER conversion)              ← job $BATCH211_JID"
echo "  batch2.1.2  (PR conversion)              ← batch2.1.1 bitince"
echo "  batch2.1.3  (HER2 conversion)            ← batch2.1.2 bitince"
echo "  batch2.1.4  (Ki67 conversion)            ← batch2.1.3 bitince"
echo "  batch2.2    (HoVerNext train 1/7)        ← batch2.1.4 bitince"
echo "  batch2.3    (HoVerNext train 2/7)        ← batch2.2 bitince"
echo "  batch2.4    (HoVerNext train 3/7)        ← batch2.3 bitince"
echo "  batch2.5    (HoVerNext train 4/7)        ← batch2.4 bitince"
echo "  batch2.6    (HoVerNext train 5/7)        ← batch2.5 bitince"
echo "  batch2.7    (HoVerNext train 6/7)        ← batch2.6 bitince"
echo "  batch2.8    (HoVerNext train 7/7 SON)    ← batch2.7 bitince"
echo ""
echo "İzleme:"
echo "  squeue -u \$USER"
echo "  tail -f patholia_b2_1_1_ER_${BATCH211_JID}.out"
echo ""
echo "Setup2 finished : $(date)"
