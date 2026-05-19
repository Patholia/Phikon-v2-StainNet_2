#!/bin/bash
#SBATCH --job-name=patholia_setup
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/prj/hpc_training/patholia/logs/setup_%j.out
#SBATCH --error=/scratch/prj/hpc_training/patholia/logs/setup_%j.err

# =============================================================
#  Patholia — MIST 1024  ·  ALL-IN-ONE SETUP
#
#  Tek seferde uçtan uca kurulum:
#    1. Repo clone (varsa atlanır)
#    2. Conda env (varsa atlanır)
#    3. PyTorch + bağımlılıklar (GPU node'da)
#    4. Phikon-v2 model cache
#    5. MIST dataset indir + aç (DOĞRU yapıda)
#    6. batch1.sh'i kuyruğa at → zincirleme başlasın
#
#  KCL CREATE'e nasıl ulaştırırsın?
#    1) Bilgisayarında bu repoyu klonladıysan, yereldeki dosyayı
#       scp ile HPC'ye gönder:
#         scp scripts/hpc/setup.sh USER@create.kcl.ac.uk:~/
#    2) Veya HPC'de direkt GitHub'dan çek:
#         curl -O https://raw.githubusercontent.com/Patholia/Phikon-v2-StainNet_2/main/scripts/hpc/setup.sh
#
#  Sonra HPC terminalinde:
#    mkdir -p /scratch/prj/hpc_training/patholia/logs
#    sbatch setup.sh
#
#  ~45-60 dk sürer. Bittiğinde batch1.sh otomatik kuyruğa girer.
# =============================================================

set -e

echo "=========================================================="
echo "  PATHOLIA SETUP — $(date)"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $(hostname)"
echo "=========================================================="

# ── Modules ──────────────────────────────────────────────────
module load anaconda3/2022.10-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0

mkdir -p /scratch/prj/hpc_training/patholia/logs

# ── 0. Work directory ────────────────────────────────────────
export WORK=/scratch/prj/hpc_training/patholia/patholia_unistain
echo "[setup] WORK = $WORK"

mkdir -p "$WORK"
cd "$WORK"

# ── 1. Clone repo (idempotent) ───────────────────────────────
if [ ! -d "$WORK/code/.git" ]; then
    echo "[setup] Cloning repository..."
    git clone https://github.com/Patholia/Phikon-v2-StainNet_2.git "$WORK/code"
else
    echo "[setup] Repo already cloned — pulling latest..."
    (cd "$WORK/code" && git pull --ff-only) || echo "[setup] git pull failed (continuing)"
fi

ls "$WORK/code" | head -5

# ── 2. Conda environment ─────────────────────────────────────
mkdir -p "$WORK/.conda_pkgs"
export CONDA_PKGS_DIRS="$WORK/.conda_pkgs"

ENV_PATH="$WORK/.conda_envs/patholia_unistain"
if [ ! -d "$ENV_PATH" ]; then
    echo "[setup] Creating conda environment..."
    conda create -p "$ENV_PATH" python=3.10 -y
else
    echo "[setup] Conda environment exists — skipping."
fi

source activate "$ENV_PATH"
echo "[setup] Python: $(python --version)"

# ── 3. Install packages (GPU node — torch ile uyumlu CUDA seçilir) ──
if ! command -v nvcc &>/dev/null; then
    echo "[setup] ERROR: nvcc not found. Setup script must run on a GPU node (it does, via SBATCH gpu partition)."
    exit 1
fi

CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
echo "[setup] Detected CUDA $CUDA_VER"

pip install --upgrade pip

if [ "$MAJOR" -ge 12 ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

pip install -r "$WORK/code/requirements.txt"
pip install -e "$WORK/code"
pip install gdown

# Verify
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[setup] CUDA OK:', torch.cuda.get_device_name(0))"
python -c "import pytorch_lightning, timm, transformers, lpips, torchmetrics; print('[setup] All packages OK')"

# ── 4. Pre-download Phikon-v2 ────────────────────────────────
export HF_HOME="$WORK/.hf_cache_patholia"
mkdir -p "$HF_HOME"

echo "[setup] Caching Phikon-v2 weights..."
python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('owkin/phikon-v2')
n = sum(p.numel() for p in m.parameters())
print(f'[setup] Phikon-v2 cached — {n:,} params')
"

# ── 5. Download MIST dataset ─────────────────────────────────
#  ÖNEMLİ: Zip dosyaları zaten HER2/TrainValAB/... yapısıyla geliyor.
#  Per-stain alt klasör oluşturursak çift iç içe geçer (MIST/HER2/HER2/...)
#  ve kod 'FileNotFoundError: ...trainA' atar. Bu yüzden direkt
#  $WORK/MIST'e açıyoruz.
if [ ! -d "$WORK/MIST/HER2/TrainValAB/trainA" ]; then
    echo "[setup] Downloading MIST dataset (~20-40 min)..."
    mkdir -p "$WORK/MIST_zips" "$WORK/MIST"

    gdown --folder https://drive.google.com/drive/folders/146V99Zv1LzoHFYlXvSDhKmflIL-joo6p \
        -O "$WORK/MIST_zips"

    ZIP_DIR=$(find "$WORK/MIST_zips" -name "HER2.zip" -exec dirname {} \; | head -1)

    if [ -z "$ZIP_DIR" ]; then
        echo "[setup] ERROR: HER2.zip not found after download — check Drive sharing permissions."
        exit 1
    fi

    for STAIN in HER2 Ki67 ER PR; do
        echo "[setup] Extracting $STAIN..."
        unzip -q "$ZIP_DIR/$STAIN.zip" -d "$WORK/MIST"
    done

    rm -rf "$WORK/MIST_zips"
    echo "[setup] MIST dataset ready."
else
    echo "[setup] MIST dataset already present — skipping download."
fi

# Verify dataset structure
echo "[setup] Verifying dataset structure..."
MISSING=0
for STAIN in HER2 Ki67 ER PR; do
    for split in trainA trainB valA valB; do
        path="$WORK/MIST/$STAIN/TrainValAB/$split"
        if [ -d "$path" ]; then
            count=$(ls "$path" 2>/dev/null | wc -l)
            printf "  %-6s %-7s : %5d files\n" "$STAIN" "$split" "$count"
        else
            echo "  MISSING: $path"
            MISSING=$((MISSING + 1))
        fi
    done
done

if [ "$MISSING" -gt 0 ]; then
    echo "[setup] ERROR: $MISSING expected directories missing — dataset extraction failed."
    echo "        Check: ls $WORK/MIST"
    exit 1
fi

# ── 6. Make batch files executable ───────────────────────────
chmod +x "$WORK/code/scripts/hpc/"batch*.sh
echo "[setup] Batch files ready: $(ls "$WORK/code/scripts/hpc/"batch*.sh | wc -l) files"

# ── 7. Submit batch1.sh — eğitim zinciri başlasın ────────────
echo ""
echo "=========================================================="
echo "  SETUP COMPLETE — eğitim zinciri başlatılıyor"
echo "=========================================================="
echo ""

BATCH1_JID=$(sbatch --parsable \
    --export=ALL,WORK="$WORK" \
    "$WORK/code/scripts/hpc/batch1.sh")

echo "[setup] batch1.sh submitted as job $BATCH1_JID"
echo ""
echo "Zincir:"
echo "  batch1 (epochs 0-20)  ← job $BATCH1_JID"
echo "  batch2 (epochs 20-40) ← batch1 bitince otomatik"
echo "  batch3 (epochs 40-60) ← batch2 bitince otomatik"
echo "  batch4 (epochs 60-80) ← batch3 bitince otomatik"
echo "  batch5 (epochs 80-100) ← batch4 bitince otomatik"
echo ""
echo "İzleme:"
echo "  squeue -u \$USER"
echo "  tail -f patholia_b1_${BATCH1_JID}.out"
echo ""
echo "İptal:"
echo "  scancel -u \$USER"
echo ""
echo "Setup finished : $(date)"
