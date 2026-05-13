# Patholia — KCL HPC Eğitim Rehberi (Başlangıç Seviyesi)

> Bu rehber Linux'a yeni başlayan biri için yazıldı. Her adım sırayla, hiçbir şey atlamadan açıklanmış. Komutları kopyala → terminalde yapıştır → Enter; başka bir şey yapman gerekmiyor.

---

## Genel Bakış — Ne Yapacağız?

KCL CREATE HPC, üniversitenin uzaktan bağlandığın güçlü bir bilgisayarı (GPU'lu). Bu rehberde:

1. HPC'ye internetten bağlanacağız (bir terminal açacağız)
2. Çalışma klasörünü hazırlayacağız
3. Kodu GitHub'dan indireceğiz
4. Python ortamını ve kütüphaneleri kuracağız
5. Modeli ve MIST veri setini indireceğiz
6. rclone'u kurup Drive'a bağlayacağız (eğitim sonu upload için)
7. Eğitimi başlatıp arka plana atacağız (3 job zincirleme, 100 epoch)
8. Eğitim **tamamen bitince** model dosyaları (best + last) otomatik olarak Drive'a yüklenecek
9. HPC'deki tüm izleri sileceğiz

**Aktif çalışman gereken süre:** ~1 saat (kurulum)
**Bekleme süresi:** GPU sıraya girene kadar 15dk–birkaç saat + eğitim 3-6 gün (otomatik)

> **Drive upload mantığı:** Her job sonunda değil, **sadece 100. epoch tamamlandığında** Drive'a yükleme yapılır. Yani zincirleme sırasında JOB1 ve JOB2 sessizce çalışır, son job (eğitim biten o job) bitince upload tetiklenir.

---

## Başlamadan Önce

Hazırlayacakların:
- KCL CREATE hesap kullanıcı adı ve şifresi (yoksa: <https://docs.er.kcl.ac.uk/CREATE/requesting_access/>)
- Bir tarayıcı (HPC portalı + Drive OAuth onayı için)
- MIST kaynak Drive klasörünün `Anyone with the link → Viewer` paylaşımına açık olması
- Drive'da eğitim sonuçlarını yazma yetkin olduğun bir klasör

Drive klasörleri:
- **Kaynak (MIST veri seti, sadece okuma):** <https://drive.google.com/drive/folders/146V99Zv1LzoHFYlXvSDhKmflIL-joo6p>
- **Hedef (eğitim sonu model dosyaları yüklenecek):** <https://drive.google.com/drive/folders/1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS>

---

## Terminal Mantığı

"Terminal" → siyah pencereli, komut yazdığın yer. Tüm kurulum tek bir HPC terminalinde yapılır:

- **Terminal 1 (HPC):** Tarayıcıda <https://portal.er.kcl.ac.uk/> → giriş yap → **Clusters → CREATE Shell** ile aç. Tüm Adım 1-19 burada.
- **Terminal 2 (HPC, isteğe bağlı):** Eğitim sırasında durumu izlemek için ikinci bir HPC terminali açabilirsin. Açmasan da olur.

**Önemli:** Terminal 1'i kapatabilirsin — sbatch ile gönderdiğin eğitim arka planda çalışmaya devam eder.

---

# 🖥️ TERMİNAL 1 — Tüm Kurulum (HPC)

## Adım 1 — HPC'ye Bağlan

Tarayıcıda <https://portal.er.kcl.ac.uk/> aç, giriş yap (MFA aktifse telefondaki kodu gir). Üst menüden **Clusters → CREATE Shell** yolunu seç. Yeni bir sekmede siyah bir pencere açılır ve şuna benzer bir satır görürsün:

```
[k1234567@erc-hpc-login3 ~]$
```

Bu **login node**. Buradan komut yazıyoruz.

> **Komut yapıştırma:** Komutu farenle seç → kopyala → terminal penceresine sağ tıkla (veya `Ctrl + Shift + V`) → Enter.

## Adım 2 — Çalışma Klasörünü Hazırla

Önce disk kontrol:
```bash
df -h /scratch/users/$USER 2>/dev/null && echo "SCRATCH VAR" || echo "SCRATCH YOK"
```

**Durum A — `SCRATCH VAR` yazıyorsa:**
```bash
export WORK=/scratch/users/$USER/patholia_unistain
```

**Durum B — `SCRATCH YOK` yazıyorsa:**
```bash
export WORK=$HOME/patholia_unistain
```

Klasörü oluştur:
```bash
mkdir -p $WORK
cd $WORK
pwd
```

`patholia_unistain` ile bitiyorsa doğru yerdesin. **`$WORK` değişkenini her yeni HPC oturumunda tekrar set etmen gerek** (sayfa sonundaki "Hatırlatma Kartı"nda blok kopyası var).

## Adım 3 — Kodu GitHub'dan İndir

```bash
git clone https://github.com/Patholia/Phikon-v2-StainNet_2.git code
cd code
ls
```

`README.md`, `requirements.txt`, `scripts`, `src` görmelisin.

## Adım 4 — Anaconda'yı Yükle

```bash
module load anaconda3
which conda
```

Hata verirse `module avail anaconda` ile doğru sürüm adını bul.

## Adım 5 — Paket Cache'ini Ayarla

```bash
mkdir -p $WORK/.conda_pkgs
export CONDA_PKGS_DIRS=$WORK/.conda_pkgs
```

## Adım 6 — Sanal Ortam Oluştur

```bash
conda create -p $WORK/.conda_envs/patholia_unistain python=3.10 -y
source activate $WORK/.conda_envs/patholia_unistain
python --version
```

`Python 3.10.x` ve prompt başında `(patholia_unistain)` görmelisin.

## Adım 7 — GPU'lu Geçici Oturuma Geç

```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --mem=32G --cpus-per-task=4 --pty bash -i
```

GPU node'a düştüğünü doğrula:
```bash
nvidia-smi
```

A100/V100/H100 + bellek görmelisin. **Bellek miktarını not et.**

Modülleri yeni shell'de tekrar yükle:
```bash
module load anaconda3
module load cuda
source activate $WORK/.conda_envs/patholia_unistain
cd $WORK/code
nvcc --version
```

`release 11.x` veya `12.x` görürsün.

## Adım 8 — PyTorch'u Kur

**CUDA 12.x ise:**
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.x ise:**
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Adım 9 — Diğer Kütüphaneler

```bash
pip install -r requirements.txt
pip install -e .
pip install gdown
```

## Adım 10 — Kurulumu Doğrula

```bash
python -c "import torch; print('CUDA çalışıyor mu?', torch.cuda.is_available())"
python -c "import pytorch_lightning, timm, transformers, lpips, torchmetrics; print('Paketler tamam')"
```

İlk komut **`True`** çıkmalı.

## Adım 11 — Phikon-v2 Modelini Önceden İndir

```bash
export HF_HOME=$WORK/.hf_cache_patholia
mkdir -p $HF_HOME
python -c "from transformers import AutoModel; m = AutoModel.from_pretrained('owkin/phikon-v2'); print('Phikon-v2 hazır')"
```

## Adım 12 — GPU Oturumundan Çık

```bash
exit
```

Prompt tekrar `erc-hpc-loginX` olmalı.

## Adım 13 — MIST Veri Setini İndir

```bash
mkdir -p $WORK/MIST_zips $WORK/MIST
gdown --folder https://drive.google.com/drive/folders/146V99Zv1LzoHFYlXvSDhKmflIL-joo6p -O $WORK/MIST_zips
ls -lh $WORK/MIST_zips
```

20-40 dk. `HER2.zip`, `Ki67.zip`, `ER.zip`, `PR.zip` görmelisin.

## Adım 14 — Zip'leri Aç

```bash
ZIP_DIR=$(find $WORK/MIST_zips -name "HER2.zip" -exec dirname {} \; | head -1)

for STAIN in HER2 Ki67 ER PR; do
  mkdir -p $WORK/MIST/$STAIN
  echo "Açılıyor: $STAIN..."
  unzip -q $ZIP_DIR/$STAIN.zip -d $WORK/MIST/$STAIN
done

rm -rf $WORK/MIST_zips
```

Doğrula:
```bash
find $WORK/MIST -maxdepth 3 -type d
```

`testA/testB` varsa `valA/valB`'ye çevir:
```bash
for s in HER2 Ki67 ER PR; do
  [ -d $WORK/MIST/$s/TrainValAB/testA ] && mv $WORK/MIST/$s/TrainValAB/testA $WORK/MIST/$s/TrainValAB/valA
  [ -d $WORK/MIST/$s/TrainValAB/testB ] && mv $WORK/MIST/$s/TrainValAB/testB $WORK/MIST/$s/TrainValAB/valB
done
```

## Adım 15 — rclone'u Kur (Drive Upload İçin)

Eğitim bitince modeli Drive'a yüklemek için rclone aracı lazım. Tek dosya:

```bash
mkdir -p $WORK/bin
cd /tmp
curl -fsSL -o rclone_patholia.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip -q rclone_patholia.zip
mv rclone-*-linux-amd64/rclone $WORK/bin/rclone
chmod +x $WORK/bin/rclone
rm -rf rclone_patholia.zip rclone-*-linux-amd64
export PATH=$WORK/bin:$PATH
rclone version
```

`rclone v1.65.0` gibi bir versiyon görmelisin.

## Adım 16 — rclone'u Drive'a Bağla

Etkileşimli setup — birkaç soruya cevap vereceksin:

```bash
rclone config --config $WORK/.rclone_patholia.conf
```

Sırayla:

| Soru | Cevap |
|------|-------|
| `e/n/d/r/c/s/q>` | `n` |
| `name>` | `patholia_gdrive` |
| `Storage>` | `drive` |
| `client_id>` | (boş, Enter) |
| `client_secret>` | (boş, Enter) |
| `scope>` | `1` |
| `service_account_file>` | (boş, Enter) |
| `Edit advanced config?` | `n` |
| `Use auto config?` | **`n`** ← önemli! |

Ekrana uzun bir komut yazacak (`rclone authorize "drive" ...`). Yapacakların:

1. Bu komutu **kopyala**
2. **Kendi bilgisayarında** (HPC'de değil) PowerShell/Terminal aç → komutu yapıştır → Enter
3. Tarayıcı açılır → **kendi gmail hesabınla** (`aysegul148ucan@gmail.com`) giriş yap, izin ver
   - ⚠️ PC'de hocanın hesabı açıksa "Use another account / Başka hesap kullan" seç → kendi hesabınla devam et. Hocanın hesabıyla onaylarsan kendi Drive klasörüne yazma yetkisi olmaz.
4. Tarayıcı uzun bir kod gösterir → kopyala
5. HPC terminaline geri dön, kodu yapıştır → Enter

Devamı:

| Soru | Cevap |
|------|-------|
| `Configure this as a Shared Drive?` | `n` |
| `Yes this is OK?` | `y` |
| `e/n/d/r/c/s/q>` | `q` |

> **Bilgisayarına rclone kuramıyorsan:** Önce <https://rclone.org/downloads/> adresinden indir, kur. Sonra adım 2'deki komutu çalıştır.
>
> **Daha kolay alternatif:** OAuth adımını telefondan da yapabilirsin. Telefonun tarayıcısında zaten kendi hesabın açık olur, hocanın hesabıyla karışmaz.

Bağlantıyı test et:
```bash
export RCLONE_CONFIG=$WORK/.rclone_patholia.conf
rclone lsd patholia_gdrive: --drive-root-folder-id 1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS
```

Hata vermezse (boş dönerse de tamamdır) bağlantı OK.

## Adım 17 — Eğitim Script'ini Oluştur

Aşağıdaki **tek bir uzun komut** — başından sonuna kopyala, yapıştır, Enter:

```bash
cat > $WORK/train_patholia.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=patholia_mist_1024
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=patholia_mist_1024_%j.out
#SBATCH --error=patholia_mist_1024_%j.err

set -e
echo "Started at $(date) on $(hostname)"

module load anaconda3
module load cuda
source activate $WORK/.conda_envs/patholia_unistain

export HF_HOME=$WORK/.hf_cache_patholia
export WANDB_MODE=offline
export PATH=$WORK/bin:$PATH
export RCLONE_CONFIG=$WORK/.rclone_patholia.conf

cd $WORK/code

CKPT_DIR=$WORK/checkpoints/mist_1024
MAX_EPOCHS=100
DRIVE_FOLDER_ID=1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS

# === Yardımcı: eğitim tamamen bittiğinde Drive'a yükle ===
upload_to_drive_if_done() {
    local epoch=$(ls $CKPT_DIR/mist_1024_epoch=*.ckpt 2>/dev/null | grep -oP 'epoch=\K[0-9]+' | sort -n | tail -1 || true)
    if [ -n "$epoch" ] && [ "$epoch" -ge $((MAX_EPOCHS - 1)) ]; then
        if [ -x "$WORK/bin/rclone" ] && [ -f "$RCLONE_CONFIG" ]; then
            echo "[upload] Eğitim tamamlandı (epoch $epoch). Drive'a yükleniyor..."
            $WORK/bin/rclone copy $CKPT_DIR patholia_gdrive: \
                --drive-root-folder-id $DRIVE_FOLDER_ID \
                --transfers 4 \
                --log-file=$WORK/rclone_patholia.log
            echo "[upload] Tamamlandı — best + last modeller Drive'da."
        else
            echo "[upload] UYARI: rclone hazır değil — manuel yükleme gerek (Adım 21)."
        fi
        return 0
    fi
    return 1
}

# === Eğitim zaten tamamlanmışsa (zincirin sonu) ===
LATEST_EPOCH=$(ls $CKPT_DIR/mist_1024_epoch=*.ckpt 2>/dev/null | grep -oP 'epoch=\K[0-9]+' | sort -n | tail -1 || true)
if [ -n "$LATEST_EPOCH" ] && [ "$LATEST_EPOCH" -ge $((MAX_EPOCHS - 1)) ]; then
    echo "[skip] Eğitim zaten tamamlanmış (epoch $LATEST_EPOCH)."
    # Drive upload daha önce başarısız olduysa burada tekrar dener (rclone copy duplicate yapmaz)
    upload_to_drive_if_done
    exit 0
fi

# === Resume mantığı ===
RESUME_ARG=""
if [ -f "$CKPT_DIR/last.ckpt" ]; then
    RESUME_ARG="--resume_from $CKPT_DIR/last.ckpt"
    echo "[resume] $CKPT_DIR/last.ckpt'ten devam ediliyor"
else
    echo "[fresh] Baştan başlanıyor"
fi

# === Eğitim ===
python scripts/train/train_mist_1024.py \
    --data_dir $WORK/MIST \
    --stains HER2 Ki67 ER PR \
    --batch_size 4 \
    --max_epochs $MAX_EPOCHS \
    --ckpt_dir $CKPT_DIR \
    --wandb_name patholia_mist_1024 \
    $RESUME_ARG

# === Eğitim bittiyse Drive'a yükle ===
upload_to_drive_if_done || echo "[info] Henüz tamamlanmadı — sonraki job devam edecek."

echo "Finished at $(date)"
EOF
```

Kontrol et:
```bash
ls -la $WORK/train_patholia.sbatch
```

> **Sbatch ne yapıyor özetle:**
> 1. Eğitim zaten 100. epoch'a ulaştıysa hemen çıkar (zincirin sonu) — bu durumda upload daha önceden başarısız olduysa tekrar denenir
> 2. `last.ckpt` varsa kaldığı yerden devam eder, yoksa baştan başlar
> 3. Eğitim yapar
> 4. Bittiğinde max_epochs'e ulaştı mı diye bakar:
>    - Ulaştıysa → **Drive'a yükle** (best + last hepsi)
>    - Ulaşmadıysa (time-limit yedi) → hiçbir şey yapmadan çıkar, sonraki job zincirden devam eder
>
> **Dosyayı değiştirmek istersen:** `nano $WORK/train_patholia.sbatch` → Çıkış: `Ctrl + X` → `Y` → Enter.

## Adım 18 — Batch Size'ı GPU'na Göre Ayarla (İsteğe Bağlı)

1024×1024'te memory yüksek. Default **`--batch_size 4`** A100 40GB'da güvenli:
- **A100 80GB:** `8` yapabilirsin (daha hızlı)
- **A100 40GB / V100 32GB:** `4` bırak
- **OOM hatası alırsan:** `2` yap

`nano $WORK/train_patholia.sbatch` ile değiştir.

## Adım 19 — Eğitimi Başlat (Otomatik Kuyruk)

**Neden zincirleme?** 1024×1024'te 100 epoch ~80-150 saat (3-6 gün) sürer, ama her job en fazla 48 saat çalışabilir. 3 job'u şimdi peş peşe kuyruğa atıyoruz — SLURM her birini öncekinin bitmesini bekleyerek otomatik başlatır.

```bash
JOB1=$(sbatch --parsable --export=ALL,WORK=$WORK $WORK/train_patholia.sbatch)
JOB2=$(sbatch --parsable --export=ALL,WORK=$WORK --dependency=afterany:$JOB1 $WORK/train_patholia.sbatch)
JOB3=$(sbatch --parsable --export=ALL,WORK=$WORK --dependency=afterany:$JOB2 $WORK/train_patholia.sbatch)
echo "Kuyruk hazır: $JOB1 → $JOB2 → $JOB3"
```

Kontrol et:
```bash
squeue -u $USER
```
Şunu görmelisin:
```
JOBID    ST  REASON
12345    R   None           ← şu an çalışıyor
12346    PD  Dependency     ← 12345 bitince başlar
12347    PD  Dependency     ← 12346 bitince başlar
```

Bu noktadan sonra **bitti.** Terminal 1'i kapatabilirsin, bilgisayarını kapatabilirsin — kuyruktaki job'lar SLURM tarafından yönetiliyor. Eğitim son job'da 100. epoch'a ulaştığında o job otomatik olarak Drive'a yükleme yapar.

> **3 job yetmezse?** Son job da bittikten sonra `sbatch --export=ALL,WORK=$WORK $WORK/train_patholia.sbatch` ile zincire ekleme yapabilirsin. Sbatch otomatik `last.ckpt`'ten devam eder; eğitim biter bitmez Drive upload kendi tetiklenir.

---

# 🖥️ TERMİNAL 2 (HPC) — İzleme (İSTEĞE BAĞLI)

İstediğin zaman HPC'ye bağlanıp eğitimin nerede olduğunu görebilirsin. Yeni oturumda önce $WORK'u set et:

```bash
export WORK=/scratch/users/$USER/patholia_unistain   # veya $HOME/patholia_unistain
cd $WORK
```

**Tüm job'ların durumu:**
```bash
squeue -u $USER
```

**Canlı log (R durumdaki job'un JOB_ID'sini yaz):**
```bash
tail -f patholia_mist_1024_<JOB_ID>.out
```
Çıkış: `Ctrl + C` (job durmaz).

**Şu ana kadar kaç epoch tamamlandı?**
```bash
ls -1 $WORK/checkpoints/mist_1024/mist_1024_epoch=*.ckpt 2>/dev/null | tail -3
```
Dosya isimlerindeki `epoch=NNN` en büyüğü = gerçek epoch sayısı. 99'a ulaşınca eğitim tamamlanmış, son job upload'u tetiklemiş demektir.

**Job'u erken durdurmak istersen:**
```bash
scancel <JOB_ID>          # tek job
scancel -u $USER          # tüm kuyruktaki job'lar
```

---

# Eğitim Bittikten Sonra

Şunlardan birini gördüğünde tamamen bitmiş demektir:
- `squeue -u $USER` artık boş çıkıyor (hiç job yok), VE
- `ls $WORK/checkpoints/mist_1024/` listesinde `mist_1024_epoch=099_step=NNNNNN.ckpt` gibi bir dosya var

**Drive'ı kontrol et:** Hedef klasörü tarayıcıda aç (<https://drive.google.com/drive/folders/1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS>). Şunları görmelisin:
- `last.ckpt` (en güncel state)
- `mist_1024_epoch=NNN_step=NNNNNN.ckpt` × 3 adet (en iyi val skorlu 3 model)

## Adım 20 — Otomatik Upload Olmadıysa (Manuel Yedek)

Eğer son job'un `.out` dosyasında `[upload] UYARI: rclone hazır değil` veya Drive'a hiç yüklenmemişse, login node'da manuel yükle:

```bash
export WORK=/scratch/users/$USER/patholia_unistain   # veya $HOME/...
export PATH=$WORK/bin:$PATH
export RCLONE_CONFIG=$WORK/.rclone_patholia.conf

rclone copy $WORK/checkpoints/mist_1024 \
    patholia_gdrive: \
    --drive-root-folder-id 1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS \
    --progress
```

`--progress` yüzde göstergesi ekler. 5-10 dakika sürer.

## Adım 21 — HPC'yi Temizle (ZORUNLU)

Drive'da dosyaların olduğunu **doğruladıktan sonra** HPC terminalinde sırayla yapıştır:

```bash
# Çalışan job'ları durdur (varsa)
scancel -u $USER

# Conda ortamından çık
conda deactivate 2>/dev/null

# Tüm çalışma klasörünü sil
rm -rf $WORK

# Home'daki cache'leri temizle
rm -rf $HOME/.cache/huggingface $HOME/.cache/torch $HOME/.cache/pip $HOME/.cache/wandb 2>/dev/null
rm -f $HOME/patholia_mist_1024_*.out $HOME/patholia_mist_1024_*.err 2>/dev/null

# Doğrulama — bir şey listelememeli
find $HOME -maxdepth 4 -name "*patholia*" 2>/dev/null
find /scratch/users/$USER -name "*patholia*" 2>/dev/null
echo "TEMİZ"
```

`TEMİZ` tek başına çıkıyorsa (öncesinde liste yoksa) tüm izler silinmiş demektir.

> **Önce Drive'da dosyaları gör, sonra sil!** Drive upload başarısız olduysa `rm -rf $WORK` ile checkpoint'ler kalıcı kaybolur.

---

# Her Yeni HPC Oturumunda — Hatırlatma Kartı

```bash
# DURUM A (scratch varsa):
export WORK=/scratch/users/$USER/patholia_unistain
# DURUM B (scratch yoksa):
# export WORK=$HOME/patholia_unistain

export HF_HOME=$WORK/.hf_cache_patholia
export RCLONE_CONFIG=$WORK/.rclone_patholia.conf
export PATH=$WORK/bin:$PATH
module load anaconda3
source activate $WORK/.conda_envs/patholia_unistain
cd $WORK/code
```

---

# Sıkça Karşılaşılan Sorunlar

| Sorun | Anlam | Çözüm |
|-------|-------|-------|
| `module: command not found` | Yeni oturum | `module load anaconda3` |
| `CUDA out of memory` | GPU belleği yetersiz | sbatch'te `--batch_size 2` yap (`nano`) |
| `torch.cuda.is_available() = False` | CUDA yüklenmedi | `module load cuda` |
| `gdown: permission denied` | Drive paylaşımı kapalı | "Anyone with the link → Viewer" |
| `FileNotFoundError: ...trainA` | Klasör yapısı yanlış | Adım 14'teki `testA → valA` adımı |
| Job sürekli `PD` | GPU kuyruğu dolu | Bekle |
| `rclone authorize` linkleri açılmıyor | PC'de rclone yok | <https://rclone.org/downloads/> indir |
| Drive'da permission denied | Yanlış hesapla OAuth | Adım 16'yı tekrar yap, kendi gmail'ini seç |
| Drive'da dosya yok ama epoch 99'a ulaştı | rclone yapılandırılmamış / network sorunu | Adım 20'deki manuel upload |
| `rclone: command not found` (yeni oturumda) | PATH ayarlanmamış | `export PATH=$WORK/bin:$PATH` |

---

# Notlar

- **Eğitim arka planda çalışır.** sbatch ile gönderdiğin job'lar SSH bağlantına bağlı değil — bilgisayarını kapatabilirsin.
- **Drive upload sadece 100. epoch tamamlandığında.** Yani JOB1 ve JOB2 sessizce çalışır (sadece checkpoint kaydederler), zincirin son job'unda eğitim biter bitmez tüm dosyalar Drive'a yüklenir.
- **rclone yapılandırılmasa bile eğitim çalışır.** Checkpoint'ler scratch'te durur, sonradan Adım 20'deki manuel komutla yüklersin.
- **Scratch'in temizlik politikası var.** Eğitim biter bitmez birkaç gün içinde Drive'ı kontrol et, sonra HPC'yi temizle (Adım 21).
- **Adlandırma:** GitHub repo = `Phikon-v2-StainNet_2`, HPC klasörü = `code`, runtime dosyaları = `patholia_*`. Temizlikte `find ... -name "*patholia*"` sadece bu projenin izlerini bulur.
