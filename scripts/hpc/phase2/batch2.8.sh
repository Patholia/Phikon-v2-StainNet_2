#!/bin/bash
#SBATCH --job-name=patholia_b2_8
#SBATCH --partition=biomed_a30_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Patholia — PHASE 2 · BATCH 2.8  (HoVerNext train chunk 7 / 7  — FINAL)
# Hedef: ~180K → 200K (toplam, son chunk)
# Sonrasında zincir yok.
set -e

TARGET_STEPS=200000

echo "=========================================================="
echo "  BATCH 2.8 (FINAL) — HoVerNext train chunk 7/7 (→ ${TARGET_STEPS} step)"
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

ls "$EXP_DIR"/*.pth 1>/dev/null 2>&1 || { echo "[batch2.8] HATA: önceki ckpt yok"; exit 1; }
LAST_CKPT=$(ls -t "$EXP_DIR"/*.pth | head -1)
echo "[batch2.8] Resume: $LAST_CKPT"

CONFIG_FILE="$WORK/checkpoints/hovernext_config_b2_8.toml"
cp "$HOVERNEXT_DIR/sample_configs/train_ihc_pannuke.toml" "$CONFIG_FILE"
sed -i "s|^data_path.*|data_path = \"$DATA_DIR\"|" "$CONFIG_FILE"
sed -i "s|^training_steps.*|training_steps = $TARGET_STEPS|" "$CONFIG_FILE"
sed -i "s|^experiment.*|experiment = \"$EXP_NAME\"|" "$CONFIG_FILE"
sed -i "s|^checkpoint_path.*|checkpoint_path = \"$LAST_CKPT\"|" "$CONFIG_FILE"

cd "$HOVERNEXT_DIR"
torchrun --standalone --nnodes=1 --nproc-per-node=1 train.py --config "$CONFIG_FILE"

echo "=========================================================="
echo "  BATCH 2.8 (FINAL) tamam — HoVerNext-IHC eğitim BİTTİ."
echo "  Final packaging başlatılıyor..."
echo "=========================================================="

# ════════════════════════════════════════════════════════════════
#   FINAL PACKAGING — proje raporu için tüm sonuçları topla
#   Strateji: dataset zaten ~70 GB, KOPYA YAPILMIYOR.
#   Mevcut $WORK/datasets/pannuke_ihc/ klasörünün İÇİNE
#   diğer sonuçlar TAŞINIYOR (mv ile). Sonra o klasör zip'leniyor.
# ════════════════════════════════════════════════════════════════

RESULTS_DIR="$WORK/datasets/pannuke_ihc"        # zaten var, dataset burada
# Zip'i daha kolay bulunur yere koy — project scratch root
RESULTS_ZIP="/scratch/prj/hpc_training/patholia/PATHOLIA_FINAL_RESULTS.zip"

# Dataset klasörünün var olduğunu doğrula
if [ ! -d "$RESULTS_DIR" ]; then
    echo "[packaging] HATA: $RESULTS_DIR yok — packaging yapılamaz!"
    exit 1
fi

# Önceki paket varsa sil
rm -f "$RESULTS_ZIP"
rm -rf "$RESULTS_DIR/logs" "$RESULTS_DIR/unistain_model" "$RESULTS_DIR/hovernext_ihc_model"
mkdir -p "$RESULTS_DIR/logs"

echo "[packaging] Container klasör: $RESULTS_DIR"
echo "[packaging] Dataset boyutu: $(du -sh $RESULTS_DIR 2>/dev/null | cut -f1)"

# ── 1. UNIStainNet checkpoint'lerini TAŞI (Phase 1) ──────────
echo "[packaging] UNIStainNet checkpoint'leri taşınıyor (mv, yer kopyası YOK)..."
if [ -d "$WORK/checkpoints/mist_1024" ]; then
    mv "$WORK/checkpoints/mist_1024" "$RESULTS_DIR/unistain_model"
    echo "  $(ls "$RESULTS_DIR/unistain_model" 2>/dev/null | wc -l) dosya taşındı"
else
    echo "  UYARI: $WORK/checkpoints/mist_1024 yok"
fi

# ── 2. HoVerNext-IHC final modelini TAŞI (Phase 2 çıktısı) ───
echo "[packaging] HoVerNext-IHC model taşınıyor (mv)..."
if [ -d "$EXP_DIR" ]; then
    mv "$EXP_DIR" "$RESULTS_DIR/hovernext_ihc_model"
    echo "  $(ls "$RESULTS_DIR/hovernext_ihc_model" 2>/dev/null | wc -l) dosya taşındı"
else
    echo "  UYARI: $EXP_DIR yok"
fi

# ── 3. Logları KOPYALA (küçük dosyalar, mv etmiyoruz ki devam edebilsin) ──
echo "[packaging] Loglar kopyalanıyor..."
cp -a $HOME/patholia_*.out "$RESULTS_DIR/logs/" 2>/dev/null
cp -a $HOME/patholia_*.err "$RESULTS_DIR/logs/" 2>/dev/null
cp -a /scratch/prj/hpc_training/patholia/logs/* "$RESULTS_DIR/logs/" 2>/dev/null
echo "  $(ls "$RESULTS_DIR/logs" 2>/dev/null | wc -l) log dosyası"

# Dataset alt-klasörlerini sayım
for M in ER PR HER2 Ki67; do
    if [ -d "$RESULTS_DIR/$M" ]; then
        n=$(ls "$RESULTS_DIR/$M/" 2>/dev/null | wc -l)
        sz=$(du -sh "$RESULTS_DIR/$M" 2>/dev/null | cut -f1)
        echo "  $M: $n dosya, $sz"
    fi
done

# ── 5. Zamanlama özeti (sacct + .out'lardan) ─────────────────
echo "[packaging] Timing summary üretiliyor..."
{
    echo "════════════════════════════════════════════════════════════════"
    echo "  PATHOLIA — ZAMANLAMA ÖZETİ"
    echo "  Oluşturuldu: $(date)"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "═══ SLURM job'ları (sacct) ═══"
    sacct -u $USER \
        --format=JobID%15,JobName%30,Elapsed,Start,End,State%15 \
        -S 2025-01-01 -P 2>/dev/null \
        | grep -iE "patholia|setup" | head -50
    echo ""
    echo "═══ Her batch'in başlangıç-bitiş zamanları (.out'lardan) ═══"
    for outf in $(ls -tr "$RESULTS_DIR/logs/"*.out 2>/dev/null); do
        bn=$(basename "$outf")
        start=$(grep -m1 -iE "Started|Started at" "$outf" 2>/dev/null | head -1)
        finish=$(grep -m1 -iE "Finished|Finished batch" "$outf" 2>/dev/null | head -1)
        echo "[$bn]"
        [ -n "$start" ] && echo "  $start"
        [ -n "$finish" ] && echo "  $finish"
        echo ""
    done
} > "$RESULTS_DIR/timing_summary.txt"

# ── 6. Eğitim metrikleri (UNIStainNet + HoVerNext) ───────────
echo "[packaging] Training metrics çıkartılıyor..."
{
    echo "════════════════════════════════════════════════════════════════"
    echo "  PATHOLIA — EĞİTİM METRİKLERİ (proje raporu için)"
    echo "  Oluşturuldu: $(date)"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "═══ Phase 1 — UNIStainNet (100 epoch, MIST 1024×1024) ═══"
    B5_OUT=$(ls -t "$RESULTS_DIR/logs/"*b5*.out 2>/dev/null | head -1)
    if [ -n "$B5_OUT" ]; then
        echo "Kaynak log: $(basename $B5_OUT)"
        echo ""
        grep -iE "Epoch 99|train/loss_g|train/loss_d|val/lpips|val/ssim|val/dab|Training complete|BATCH 5" "$B5_OUT" \
            | tail -20
    else
        echo "(Phase 1 batch5 log bulunamadı)"
    fi
    echo ""
    echo "═══ Phase 2 — HoVerNext-IHC (200K step, sentetik IHC) ═══"
    B28_OUT=$(ls -t "$RESULTS_DIR/logs/"*b2_8*.out 2>/dev/null | head -1)
    if [ -n "$B28_OUT" ]; then
        echo "Kaynak log: $(basename $B28_OUT)"
        echo ""
        grep -iE "val_loss|val_f1|val_pq|epoch|step.*loss|best|Training" "$B28_OUT" \
            | tail -30
    else
        echo "(Phase 2 batch2.8 log bulunamadı)"
    fi
    echo ""
    echo "═══ HoVerNext-IHC checkpoint dosyaları ═══"
    ls -lh "$RESULTS_DIR/hovernext_ihc_model/" 2>/dev/null
} > "$RESULTS_DIR/training_metrics.txt"

# ── 7. Tamamlanma durumu ────────────────────────────────────
echo "[packaging] Completion status üretiliyor..."
{
    echo "════════════════════════════════════════════════════════════════"
    echo "  PATHOLIA — PROJE TAMAMLANMA DURUMU"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Durum         : SUCCESS"
    echo "Phase 1 (UNIStainNet) ended : $(stat -c %y "$WORK/checkpoints/mist_1024/last.ckpt" 2>/dev/null)"
    echo "Phase 2 (HoVerNext)   ended : $(date)"
    echo ""
    echo "Final HoVerNext-IHC ckpt    : $LAST_CKPT"
    echo "Initial pretrained          : pannuke_convnextv2_tiny_3 (Zenodo)"
    echo ""
    echo "═══ Paket içeriği ═══"
    find "$RESULTS_DIR" -maxdepth 3 -type f 2>/dev/null | head -80
    echo ""
    echo "═══ Klasör boyutları ═══"
    du -sh "$RESULTS_DIR"/*/ 2>/dev/null
    echo ""
    echo "═══ Toplam boyut ═══"
    du -sh "$RESULTS_DIR" 2>/dev/null
} > "$RESULTS_DIR/completion_status.txt"

# ── 8. README ───────────────────────────────────────────────
cat > "$RESULTS_DIR/README.md" <<'EOF'
# Patholia — Final Sonuç Paketi

Bu klasör Patholia projesinin tüm sonuçlarını içeriyor. Drive'a yükleyip
proje raporu yazımında kaynak olarak kullanılacak.

## Yapı (pannuke_ihc/ klasörünün içinde her şey)

```
pannuke_ihc/                     ← container (dataset zaten burada)
├── README.md                    ← bu dosya
├── timing_summary.txt           ← SLURM süreleri + batch zamanları
├── training_metrics.txt         ← UNIStainNet + HoVerNext final metrikleri
├── completion_status.txt        ← tamamlanma durumu, dosya listesi
├── ER/   {images,labels}.npy    ← sentetik IHC veri (UNIStainNet çıktısı)
├── PR/   {images,labels}.npy
├── HER2/ {images,labels}.npy
├── Ki67/ {images,labels}.npy
├── logs/                        ← tüm batch'lerin .out ve .err dosyaları
├── unistain_model/              ← Phase 1 UNIStainNet checkpoint'leri
│   └── *.ckpt (last + top-3 best)
└── hovernext_ihc_model/         ← Phase 2 HoVerNext-IHC fine-tuned model
    └── *.pth, params.toml
```

## Kullanım (raporu yazarken)

1. `timing_summary.txt` → "Yöntem" başlığında eğitim süreleri için
2. `training_metrics.txt` → "Bulgular" başlığında doğruluk skorları için
3. `completion_status.txt` → ek/tablo olarak özet bilgi
4. `logs/` → spesifik bir hata veya milestone aranıyorsa
5. Model dosyaları → tekrarlanabilirlik için arşiv

## Drive'a yükleme

Bu klasörü olduğu gibi veya zip versiyonunu yükle.
EOF

# ── 9. ZIP (mevcut pannuke_ihc/ klasörünü zip'le, kopya yok) ──
echo "[packaging] ZIP üretiliyor — dataset büyük olduğu için 30-90 dk sürebilir..."
echo "[packaging] Kaynak: $RESULTS_DIR ($(du -sh $RESULTS_DIR 2>/dev/null | cut -f1))"
cd "$WORK/datasets"
zip -r --quiet "$RESULTS_ZIP" "pannuke_ihc/"

if [ -f "$RESULTS_ZIP" ]; then
    echo "[packaging] ZIP OK: $RESULTS_ZIP ($(du -sh $RESULTS_ZIP 2>/dev/null | cut -f1))"
else
    echo "[packaging] HATA: ZIP üretilemedi"
fi

# ── 10. Bilgi notu — patholia klasörü içine (kolay bulma) ────
#  $HOME'a dokunmuyoruz — tüm proje izi /scratch/.../patholia/ altında.
cat > "/scratch/prj/hpc_training/patholia/WHERE_IS_RESULTS.txt" <<EOF
═══════════════════════════════════════════════════════════════
  PATHOLIA — Sonuç dosyaları nerede?
  Oluşturuldu: $(date)
═══════════════════════════════════════════════════════════════

ZIP (tek dosya, scp/Drive için en pratik):
    $RESULTS_ZIP

KLASÖR (zip'in açılmış hâli, browse etmek için):
    $RESULTS_DIR

İçerik:
    - Sentetik IHC dataset (ER, PR, HER2, Ki67)
    - UNIStainNet checkpoint'leri (Phase 1)
    - HoVerNext-IHC fine-tuned model (Phase 2)
    - Tüm batch logları
    - timing_summary.txt, training_metrics.txt, completion_status.txt

PC'ye çekmek için (kendi bilgisayarından):
    scp <KCL_USER>@create.kcl.ac.uk:$RESULTS_ZIP ~/Downloads/

Temizlik (proje bittiğinde):
    rm -rf /scratch/prj/hpc_training/patholia/
    rm -f \$HOME/patholia_*.out \$HOME/patholia_*.err
═══════════════════════════════════════════════════════════════
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  TÜM SÜREÇ BAŞARIYLA TAMAMLANDI."
echo ""
echo "  Drive'a yüklenecek (sadece BU iki şeyden biri):"
echo "    Klasör  : $RESULTS_DIR"
echo "    Zip     : $RESULTS_ZIP"
echo ""
echo "  Zip yolunu hatırlatan dosya (patholia içinde):"
echo "    /scratch/prj/hpc_training/patholia/WHERE_IS_RESULTS.txt"
echo ""
echo "  PC'ye scp ile çekmek için:"
echo "    scp <KCL_USER>@create.kcl.ac.uk:$RESULTS_ZIP ./patholia_final.zip"
echo ""
echo "  TEMİZLİK (proje bittikten sonra):"
echo "    rm -rf /scratch/prj/hpc_training/patholia/"
echo "    rm -f  \$HOME/patholia_*.out \$HOME/patholia_*.err"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Finished batch 2.8 (FINAL) : $(date)"
