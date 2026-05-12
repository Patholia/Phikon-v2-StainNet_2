# Patholia-UNIStain — KCL CREATE HPC üzerinde MIST eğitimi
**Adım adım not defteri** · İz bırakmayan kurulum + eğitim + temizlik

> Bu not defteri, **King's College London CREATE HPC**'de ("sanal bilgisayar" — A100 GPU'lu) geçici bir conda environment kurup MIST datasetiyle UNIStainNet eğitmek için yazıldı. **Tüm iş HPC içinde yapılır.** Env, dataset, model, checkpoint — hepsi HPC'nin scratch alanında oluşur, sonunda tek komutla silinir.
>
> **Önemli — silme güvenliği:** HPC hesabı paylaşılıyor olabilir (hoca'nın hesabı). Tüm dosya/env adlarına **`patholia_unistain`** önekini koyduk. Böylece temizlik adımında:
> ```bash
> find $HOME /scratch/users/$USER -name "*patholia*"   # bu projeden tüm izleri listeler
> ```
> komutuyla **sadece bu projenin izlerini** görüp silebilirsin. Hoca'nın başka işleri etkilenmez.
>
> **Notebook'taki tüm komutlar HPC terminalinde çalışacak.** HPC'ye nasıl erişiyorsan (tarayıcı portal, SSH istemcisi, vs.) sadece bir terminal açıp komutları yapıştır.

---

## 0. Genel Akış (özet)

| Aşama | Yapılan iş | Süre |
|------|-----------|------|
| 1 | KCL hesabı + Drive paylaşımı (bir kez) | ~10 dk |
| 2 | HPC terminaline gir | 1 dk |
| 3 | Çalışma klasörü + scratch alanı seçimi | 2 dk |
| 4 | Proje kodunu HPC'ye taşıma (git clone) | 2 dk |
| 5 | Anaconda modülü + conda env oluşturma | 5 dk |
| 6 | Bağımlılıkları kurma (interactive GPU node'da) | 15-25 dk |
| 7 | Foundation model (**Phikon-v2**, açık erişim) — HF cache scratch'e + ön-indirme | 5 dk |
| 8 | Drive'dan MIST datasetini indirme + açma | 20-40 dk |
| 9 | Veri yapısını doğrulama | 2 dk |
| 10 | SLURM batch job ile eğitim başlatma (job bitince auto-upload yapar) | 1 dk |
| 11 | Eğitim takibi | sürekli |
| 12 | **Sonuçları Drive'a yükleme (rclone) — sbatch öncesi setup + (gerekirse) manuel** | 10 dk |
| 13 | **TEMİZLİK — her şeyi silme** | 2 dk |

**Drive linkleri (sabitlenmiş):**
- Source (MIST datasetı): `https://drive.google.com/drive/folders/146V99Zv1LzoHFYlXvSDhKmflIL-joo6p` → folder ID = `146V99Zv1LzoHFYlXvSDhKmflIL-joo6p`
- Destination (eğitim sonu checkpoint'leri): `https://drive.google.com/drive/folders/1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS` → folder ID = `1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS`

---

## 1. Ön Hazırlık (bir kez)

### 1.1  KCL CREATE hesabı + erişim
Hoca'nın öğrenci hesabıyla KCL CREATE'e erişimi olmalı. Yoksa: `https://docs.er.kcl.ac.uk/CREATE/requesting_access/` üzerinden başvur, onay e-postası gelene kadar bekle.

### 1.2  Drive paylaşım ayarlarını kontrol et
**Source klasörü** (`146V99Zv1LzoHFYlXvSDhKmflIL-joo6p`) için tarayıcıda:
1. Klasör linkini aç → sağ üstte **Share** ikonu
2. "General access" → **Anyone with the link** → **Viewer** → Done
3. İçindeki HER2/Ki67/ER/PR zipleri otomatik viewer izniyle gelir → `gdown --folder` ile direk inecek.

**Destination klasörü** (`1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS`) senin/hoca'nın Drive'ında — ek paylaşım gerekmez. rclone OAuth'u onayladığında yazma yetkisi alır (12. bölüm).

---

## 2. HPC Terminaline Giriş

KCL CREATE'e nasıl erişiyorsan (örn. tarayıcı tabanlı portal terminali — `https://portal.er.kcl.ac.uk/` veya SSH istemcisi) **bir terminal aç ve buradan itibaren tüm komutları orada çalıştır.**

İlk SSH ise: portalda MFA'yı aktif et (`https://portal.er.kcl.ac.uk/mfa/`), sonra bağlan.

Login node'lardan birine düşersin: `erc-hpc-login3` veya `erc-hpc-login4`. Aşağıdaki tüm komutlar burada (login node) ve daha sonra job içinde (compute node) çalışacak.

> **Uyarı:** Login node'da **hiçbir zaman** ağır iş çalıştırma. İndirme, küçük dosya işleri, sbatch göndermek için kullan. Pytorch kurulumu bile interactive node'a geçmek isteyebilir (6. bölüm).

---

## 3. Çalışma Alanı

KCL CREATE'te home dizini kısıtlı kotaya sahip, datasetler için **scratch** kullanılması önerilir. Kullanıcı scratch alanın genelde:
```
/scratch/users/$USER
```
biçimindedir. Önce kotayı kontrol et:
```bash
quota
df -h /scratch/users/$USER 2>/dev/null || echo "scratch yok, home kullanacağız"
```

Çalışma klasörünü oluştur (scratch varsa orada, yoksa $HOME altında). **Klasör adı `patholia_unistain` → tüm proje izlerini bu isimle bulup silebileceğiz:**
```bash
# scratch varsa:
export WORK=/scratch/users/$USER/patholia_unistain
# yoksa:
# export WORK=$HOME/patholia_unistain

mkdir -p $WORK
cd $WORK
```

`echo "export WORK=$WORK" >> ~/.bashrc` **yapma** → işimiz bitince $HOME'daki bashrc'de iz kalsın istemiyoruz. Her oturumda `export WORK=...` yazarız.

---

## 4. Proje Kodunu HPC'ye Taşıma

Kod GitHub'da olsun (yoksa hocadan repo linkini iste). Direkt git ile çekiyoruz:
```bash
cd $WORK
git clone https://github.com/<KULLANICI>/UNIStainNet.git code
cd code
ls   # src/, scripts/, requirements.txt görmeli
```

> GitHub değil de başka bir yerden indirmen gerekiyorsa (Google Drive, kişisel sunucu, vb.) bana söyle, ona göre adım yazarım.

---

## 5. Conda Env Oluşturma

KCL CREATE'te **Anaconda module** var.
```bash
module avail anaconda 2>&1 | head
module load anaconda3
which conda
```

Default conda paket cache'i $HOME'da olur ve home kotanı doldurur. Cache'i de scratch'e taşıyalım:
```bash
mkdir -p $WORK/.conda_pkgs $WORK/.conda_envs
export CONDA_PKGS_DIRS=$WORK/.conda_pkgs
```

Env'i `-p` ile **path bazlı** oluştur (home altındaki conda envs listesine eklemez, iz bırakmaz). Env adı: `patholia_unistain`:
```bash
conda create -p $WORK/.conda_envs/patholia_unistain python=3.10 -y
```

Aktifleştir:
```bash
conda activate $WORK/.conda_envs/patholia_unistain
python --version   # 3.10.x göstermeli
```

> **İpucu:** `conda init` çalıştırma. Aksi takdirde `~/.bashrc` dosyasına satırlar yazılır ve "iz bırakma" hedefin bozulur. Sadece `module load anaconda3 && source activate $WORK/.conda_envs/patholia_unistain` ile manuel aktive et.

---

## 6. Bağımlılıkları Kurma (Interactive GPU Node'da)

Login node'da pip install zorlanabilir (bellek + ağ). 1 saatlik bir GPU oturumu açıp orada kuralım:
```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --mem=32G --cpus-per-task=4 --pty bash -i
```

Compute node'a düştün. Modülleri ve env'i tekrar yükle (interactive shell yeni başladı):
```bash
module load anaconda3
module load cuda
source activate $WORK/.conda_envs/patholia_unistain
cd $WORK/code

nvidia-smi   # GPU görünüyor mu? (A100, V100, H100 olabilir)
nvcc --version
```

> **CUDA neden lazım?** A100 fiziksel GPU; PyTorch'un onunla konuşabilmesi için **CUDA toolkit** (sürücü + cuDNN + cuBLAS) aktif olmalı. `module load cuda` bunu env path'ine ekliyor. Aktif değilse `torch.cuda.is_available()` False döner, eğitim CPU'da kalır (haftalarca sürer). Yani CUDA "alakasız" değil, **zorunlu altyapı**.

### 6.1  PyTorch (CUDA sürümüne uygun)
Cluster'da CUDA 12.x varsa:
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
CUDA 11.8 ise:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
`nvcc --version` çıktısındaki major sürüme bak ve uygun olanı seç.

### 6.2  Geri kalan paketler
```bash
pip install -r requirements.txt
pip install -e .
pip install gdown            # Drive indirme için
```
> `transformers` zaten `requirements.txt`'ta — Phikon-v2 modelini yüklemek için kullanılacak.

### 6.3  Kurulumu doğrula
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "import pytorch_lightning, timm, transformers, lpips, torchmetrics; print('OK')"
python -c "from transformers import AutoModel; print('transformers import OK')"
```

Üçü de hatasız çıkmalı. Hata varsa: pip versiyonunu kontrol et, CUDA toolkit'i tekrar yükle.

### 6.4  Interactive oturumdan çık (gerek kalmadıysa)
İndirme adımı için login node'da kalmak daha mantıklı. `exit` yazarak çık:
```bash
exit
```
Login node'a geri dönersin.

---

## 7. Foundation Model (Phikon-v2 — açık erişim)

Projede **UNI** yerine **Phikon-v2** (`owkin/phikon-v2`) kullanılıyor. Phikon-v2:
- UNI ile **aynı mimari** (ViT-L/16), aynı patch size (16), aynı embedding dim (1024)
- **Gating yok** — HuggingFace hesabı veya token gerektirmez
- 303M parametre (~1.2 GB), ImageNet normalize ile birebir aynı kullanım

### 7.1  HF cache'ini scratch'e yönlendir (home kotanı koru + silmesi kolay)
Default `$HOME/.cache/huggingface/` ~1.2 GB yer kaplar ve home kotanı kemirir. Cache'i scratch altında **`patholia` izli** bir klasöre taşı:
```bash
export HF_HOME=$WORK/.hf_cache_patholia
mkdir -p $HF_HOME
```
Bu satırı **`train_patholia.sbatch` script'in başına** da ekleyeceğiz (10.1'de örnek var).

### 7.2  Modeli önceden indir (önerilen)
Compute node'da internet kısıtlı olabilir. Login node'da önce cache'le:
```bash
module load anaconda3
source activate $WORK/.conda_envs/patholia_unistain
export HF_HOME=$WORK/.hf_cache_patholia

python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('owkin/phikon-v2')
print('Phikon-v2 cached OK | params:', sum(p.numel() for p in m.parameters()))
"
```

> `huggingface-cli login` **gerekmez**. Token, onay, gated model erişimi yok — direkt indirilir.

> Cache yolu artık: `$WORK/.hf_cache_patholia/hub/models--owkin--phikon-v2/` — temizlik adımında `$WORK` ile birlikte silinecek.

---

## 8. Drive'dan MIST Datasetini İndirme

Login node'da (internet var):
```bash
mkdir -p $WORK/MIST_zips $WORK/MIST
cd $WORK/MIST_zips
```

**Tek komutla tüm klasörü indir** — `gdown --folder` source klasörünün içindeki 4 zip'i otomatik çeker:
```bash
gdown --folder https://drive.google.com/drive/folders/146V99Zv1LzoHFYlXvSDhKmflIL-joo6p -O $WORK/MIST_zips
```

> **Not:** `gdown --folder` 50 dosyaya kadar otomatik indirir; 4 zip için bol bol yeter.
> **gdown takılırsa:** "permission denied" hatası → Drive klasör paylaşımı "Anyone with the link" değil demektir (1.2'ye dön).
> **Çok büyükse:** Drive büyük dosyalar için "virus scan warning" verir. gdown 5.x bunu otomatik bypass eder. Eski sürüm varsa `pip install -U gdown` ile güncelle.

Boyutları kontrol et:
```bash
ls -lh $WORK/MIST_zips
# HER2.zip, Ki67.zip, ER.zip, PR.zip görmeli (içeride alt klasör olabilir)
find $WORK/MIST_zips -name "*.zip" -exec ls -lh {} \;
```

### 8.1  Zipleri aç ve klasör yapısını oluştur
Zip içinde direkt `TrainValAB/{trainA,trainB,valA,valB}/` olduğu için doğrudan **`$WORK/MIST/<STAIN>/`** altına açıyoruz. Tek loop ile 4 stain hallolur:

```bash
ZIP_DIR=$(find $WORK/MIST_zips -name "HER2.zip" -exec dirname {} \; | head -1)
echo "Zipler burada: $ZIP_DIR"

for STAIN in HER2 Ki67 ER PR; do
  mkdir -p $WORK/MIST/$STAIN
  echo "Açılıyor: $STAIN.zip → $WORK/MIST/$STAIN/"
  unzip -q $ZIP_DIR/$STAIN.zip -d $WORK/MIST/$STAIN
done
```

Sonuç olarak şu yapı oluşmalı:
```
$WORK/MIST/HER2/TrainValAB/{trainA,trainB,valA,valB}/
$WORK/MIST/Ki67/TrainValAB/{...}/
$WORK/MIST/ER/TrainValAB/{...}/
$WORK/MIST/PR/TrainValAB/{...}/
```

### 8.2  Zipleri sil (yer aç)
```bash
rm -rf $WORK/MIST_zips
```

---

## 9. Veri Yapısını Doğrulama

```bash
tree -L 3 $WORK/MIST 2>/dev/null || find $WORK/MIST -maxdepth 3 -type d
```

Sayım kontrolü:
```bash
for s in HER2 Ki67 ER PR; do
  echo "=== $s ==="
  for d in trainA trainB valA valB; do
    n=$(ls $WORK/MIST/$s/TrainValAB/$d 2>/dev/null | wc -l)
    echo "  $d: $n dosya"
  done
done
```

Eğer `valA/valB` yoksa (datasette `testA/testB` olarak gelmiş olabilir):
```bash
# Test'i val olarak yeniden adlandır:
for s in HER2 Ki67 ER PR; do
  [ -d $WORK/MIST/$s/TrainValAB/testA ] && mv $WORK/MIST/$s/TrainValAB/testA $WORK/MIST/$s/TrainValAB/valA
  [ -d $WORK/MIST/$s/TrainValAB/testB ] && mv $WORK/MIST/$s/TrainValAB/testB $WORK/MIST/$s/TrainValAB/valB
done
```

> Dataset yapısı README'deki şablonla **aynı** olmalı, yoksa `src/data/mist_dataset.py` `FileNotFoundError` atar.

---

## 10. SLURM Batch Job ile Eğitim

Eğitim uzun sürecek (büyük olasılıkla saatler/günler). Interactive değil, **sbatch** ile arka planda çalıştırırız.

> **ÖNEMLİ — sbatch göndermeden önce rclone'u yapılandır:** Aşağıdaki sbatch script'inin sonunda eğitim biter bitmez checkpoint'leri Drive'a yükleyen bir blok var. Bu bloğun çalışabilmesi için **12.1 (rclone kurulumu)** ve **12.2 (Drive OAuth)** adımlarını **şimdi yap**, sonra geri dönüp `sbatch` ile job'u gönder. rclone yapılandırılmamışsa eğitim yine çalışır ama upload sessizce atlanır → 12.3'teki komutla manuel yüklersin.

### 10.1  Batch script'i oluştur
```bash
cat > $WORK/train_patholia.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=patholia_unistain_mist
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00          # 2 gün; gerekiyorsa artır
#SBATCH --output=patholia_unistain_%j.out
#SBATCH --error=patholia_unistain_%j.err

set -e
echo "Started at $(date) on $(hostname)"

module load anaconda3
module load cuda
source activate $WORK/.conda_envs/patholia_unistain

# HF cache'i scratch'te tut (home kotasını koru)
export HF_HOME=$WORK/.hf_cache_patholia

# wandb hesabın yoksa kayıt yapmasını engelle:
export WANDB_MODE=offline
# Veya tamamen kapat:
# export WANDB_MODE=disabled

cd $WORK/code

python scripts/train/train_mist.py \
    --data_dir $WORK/MIST \
    --stains HER2 Ki67 ER PR \
    --batch_size 8 \
    --max_epochs 75 \
    --ckpt_dir $WORK/checkpoints/patholia_unistain_mist \
    --wandb_name patholia_unistain_mist_run1

# === Eğitim bittikten sonra checkpoint'leri otomatik olarak Drive'a yükle ===
# Bu blok ancak rclone önceden 12.1-12.2'deki gibi yapılandırıldıysa çalışır.
# Aksi takdirde sessizce atlanır → manuel yüklemek için 12.3'teki komutu kullan.
export PATH=$WORK/bin:$PATH
export RCLONE_CONFIG=$WORK/.rclone_patholia.conf

if [ -x "$WORK/bin/rclone" ] && [ -f "$RCLONE_CONFIG" ]; then
    echo "[upload] Drive'a yükleniyor → folder 1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS"
    $WORK/bin/rclone copy $WORK/checkpoints/patholia_unistain_mist \
        patholia_gdrive: \
        --drive-root-folder-id 1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS \
        --transfers 4 \
        --log-file=$WORK/rclone_patholia.log \
        --log-level INFO
    echo "[upload] Tamamlandı (log: $WORK/rclone_patholia.log)"
else
    echo "[upload] UYARI: rclone kurulu/yapılandırılmış değil — upload atlandı."
    echo "[upload]        Manuel yüklemek için 12.3'teki komutu çalıştır."
fi

echo "Finished at $(date)"
EOF
```

> **Batch_size notu:** A100 80GB ise 16, A100 40GB / V100 ise 8 kullan. `nvidia-smi` ile düşürdüğün node'un GPU bellek miktarını gör, ona göre `--batch_size`'ı ayarla.
> **Partition:** İlk denemede `gpu` queue dolu ise `interruptible_gpu` daha kısa kuyruk olabilir ama job kesilebilir (checkpoint kullanıldığı için kaldığı yerden devam eder).

### 10.2  Job'u gönder
```bash
# WORK env'inin script görünür olması için:
export WORK=$WORK    # zaten set
sbatch --export=ALL,WORK=$WORK $WORK/train_patholia.sbatch
```

Çıktı: `Submitted batch job 12345678` → bu **JOB_ID**'yi not al.

---

## 11. Eğitimi İzleme

### 11.1  Kuyruktaki durumu gör
```bash
squeue -u $USER
```
Status kodu:
- `PD` = pending (sıraya alındı)
- `R` = running
- `CG` = completing

### 11.2  Canlı log
```bash
tail -f patholia_unistain_<JOB_ID>.out
# Çıkış: Ctrl+C (job durmaz, sadece log takibi durur)
```

### 11.3  GPU/RAM kullanımı
Job çalışırken, başka SSH oturumunda:
```bash
srun --jobid <JOB_ID> --pty bash -c 'nvidia-smi'
```

### 11.4  Job'u durdurma
```bash
scancel <JOB_ID>
```

### 11.5  Wandb offline modunda log'ları sonradan senkronla (opsiyonel)
Eğitim bittiğinde, hesabın varsa:
```bash
wandb login
wandb sync $WORK/code/wandb/offline-run-*
```

---

## 12. Sonuçları Drive'a Yükleme (rclone)

Eğitim bitince `$WORK/checkpoints/patholia_unistain_mist/` altında `.ckpt` dosyaları olacak (`save_top_k=3` + `last.ckpt`). Bunları **rclone** ile Drive'daki destination klasörüne (`1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS`) yükleyeceğiz.

### 12.1  rclone'u HPC'ye kur (login node, internet var)
rclone tek binary, conda gerektirmez. **$WORK/bin altına kuruyoruz** → temizlikte $WORK ile birlikte gidiyor:
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

### 12.2  Drive remote'u yapılandır (headless OAuth)
HPC compute node'unda tarayıcı yok, OAuth iki adımda yapılır: HPC bir kod üretir, sen onu **tarayıcı açabildiğin herhangi bir yerde** (örn. KCL portal sayfasını açtığın aynı tarayıcı, telefon, herhangi bir bilgisayar) onaylarsın.

**HPC terminalinde:**
```bash
rclone config --config $WORK/.rclone_patholia.conf
```
> `--config` ile özel config yolu veriyoruz → token `~/.config/rclone/` altına yazılmaz, $WORK'ta kalır. Temizlik kolay.

Sırayla şunları yaz/seç:
- `n` (new remote)
- name: `patholia_gdrive`
- Storage: `drive` (sayısı listede görünür — Google Drive)
- `client_id`: boş bırak (Enter)
- `client_secret`: boş bırak (Enter)
- `scope`: `1` (drive, full access)
- `service_account_file`: boş bırak (Enter)
- `Edit advanced config?` → `n`
- `Use auto config?` → **`n`** (önemli — headless mod)
- Ekrana bir komut/URL yazacak → kopyala, tarayıcıda aç, `aysegul148ucan@gmail.com` ile giriş yap, izin ver, çıkan kodu HPC terminaline yapıştır
- `Configure this as a Shared Drive?` → `n`
- `Yes this is OK` → `y`
- `q` (quit)

### 12.3  Yüklemeyi yap
```bash
export PATH=$WORK/bin:$PATH
export RCLONE_CONFIG=$WORK/.rclone_patholia.conf

# Test: destination klasörünü listele (boş veya mevcut içeriği gösterir)
rclone lsd patholia_gdrive: --drive-root-folder-id 1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS

# Checkpoint'leri yükle (progress bar ile)
rclone copy $WORK/checkpoints/patholia_unistain_mist \
    patholia_gdrive: \
    --drive-root-folder-id 1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS \
    --progress \
    --transfers 4

# Doğrulama
rclone ls patholia_gdrive: --drive-root-folder-id 1FNdkkrOZN9Zvw3kEmp3BXKxDbHGGp2cS
```

`.ckpt` dosyaları genelde 200MB-1GB arası olur, yükleme birkaç dakika sürebilir.

### 12.4  Otomatik yükleme (sbatch'e zaten eklendi)
Otomatik upload bloğu **10.1'deki sbatch script'inin içinde** hazır — `python scripts/train/train_mist.py …` satırından hemen sonra `rclone copy …` bloğu var. Eğitim biter bitmez Drive'a yüklenir.

> **Şart:** 12.1 (rclone binary) ve 12.2 (Drive OAuth) **sbatch göndermeden önce** tamamlanmış olmalı, yoksa upload bloğu sessizce atlanır (sbatch log'unda `[upload] UYARI` görürsün) ve checkpoint'leri 12.3'le manuel yüklemen gerekir.
>
> **Compute node'da internet yoksa:** rclone upload başarısız olabilir → job bitince login node'a dönüp 12.3 komutunu manuel çalıştır.

---

## 13. TEMİZLİK — Her şeyi silme (zorunlu adım)

İşin bittiyse veya yarıda kestiysen HPC terminalinde sırayla çalıştır:

```bash
# 13.1 Çalışan job'ları durdur
scancel -u $USER

# 13.2 Conda env'i deaktive et ve sil
conda deactivate 2>/dev/null || true
conda env remove -p $WORK/.conda_envs/patholia_unistain -y 2>/dev/null || rm -rf $WORK/.conda_envs

# 13.3 Tüm scratch çalışma klasörünü sil
# (kod, dataset, checkpoint, rclone binary, HF cache, rclone config — hepsi gider)
rm -rf $WORK

# 13.4 Home'daki cache'leri temizle (HF cache zaten $WORK'ta ama emniyet için)
rm -rf $HOME/.cache/huggingface
rm -rf $HOME/.cache/torch
rm -rf $HOME/.cache/pip
rm -rf $HOME/.cache/wandb
rm -rf $HOME/.conda      # conda init çalıştırmadıysan zaten yok
rm -rf $HOME/wandb

# 13.5 Doğrulama — "patholia" izli hiçbir şey kalmamalı
find $HOME -maxdepth 4 -name "*patholia*" 2>/dev/null
find /scratch/users/$USER -name "*patholia*" 2>/dev/null
ls $WORK 2>/dev/null         # "No such file or directory" çıkmalı
squeue -u $USER              # boş olmalı
echo "HPC TEMİZ"
```

**Sonuç:** HPC'de tüm env, dataset, checkpoint, cache silindi. `find ... patholia*` boş çıkarsa hiçbir iz kalmamış demektir — hoca'nın hesabındaki diğer projeleri/dosyaları etkilenmedi.

---

## Sorun Giderme

| Sorun | Çözüm |
|------|------|
| `module: command not found` | Login node'da değilsin veya `.bashrc` kirli → tekrar `module load` yap |
| `CUDA out of memory` | `--batch_size 4` veya 2 yap; `--cpus-per-task` artırma değil **GPU memory** sorunu |
| `torch.cuda.is_available() == False` | `module load cuda` yapmadın veya CUDA toolkit env'le uyumsuz → 6.1'deki PyTorch CUDA sürümünü kontrol et |
| `gdown: permission denied` | Drive'da paylaşım "Anyone with the link" değil |
| `FileNotFoundError: ...trainA` | Adım 9'daki klasör yapısını yeniden kontrol et |
| `Access to model owkin/phikon-v2 is restricted` | Phikon-v2 normalde açık erişim → ağ/proxy kısıtı olabilir, login node'dan tekrar dene |
| `ModuleNotFoundError: transformers` | `pip install -r requirements.txt` çalışmadı veya yanlış env aktif — env'i kontrol et |
| `OSError: Can't load tokenizer` Phikon-v2 yüklerken | İnternete erişim yok (compute node) → login node'da 7.2'deki ön-indirme adımını yap, sonra job'u tekrar başlat |
| `Disk quota exceeded` | `$HOME` doldu → cache'leri sil, conda env'i scratch'e taşı |
| Job sürekli `PD` (pending) | Kuyruk dolu, `interruptible_gpu` partition'ını dene veya `--time`'ı azalt |
| `rclone: command not found` (yeni oturumda) | `export PATH=$WORK/bin:$PATH` yapmayı unutma |
| rclone "couldn't find folder" | `--drive-root-folder-id` parametresine doğru folder ID verdiğinden emin ol |
| rclone OAuth "expired token" | `rclone config reconnect patholia_gdrive: --config $WORK/.rclone_patholia.conf` ile yeniden authorize et |

---

## Notlar / Hatırlatmalar

- Her yeni SSH oturumunda en başta:
  ```bash
  export WORK=/scratch/users/$USER/patholia_unistain
  export HF_HOME=$WORK/.hf_cache_patholia
  export RCLONE_CONFIG=$WORK/.rclone_patholia.conf
  export PATH=$WORK/bin:$PATH
  module load anaconda3
  source activate $WORK/.conda_envs/patholia_unistain
  ```
- Login node'da heavy iş yapma → `srun --pty` ile interactive node'a geç.
- `sbatch` ile başlattığın job, SSH oturumunu kapatsan da çalışmaya devam eder.
- Checkpoint'ler `--ckpt_dir`'e yazılır → `save_top_k=3` ayarı en iyi 3 model + `last.ckpt`'i tutar.
- **Sıralama önemli:** Eğitim bitince önce 12 (Drive'a yükle) → 13 (temizlik). Temizliği atlasan checkpoint'ler scratch'te kalır ve auto-purge'de gidebilir.
- **Adlandırma güvenliği:** Tüm dosya/env/cache adlarında **`patholia`** önekini koruduk. Temizlikte `find ... -name "*patholia*"` ile sadece bu projenin izlerini görürsün → hoca'nın başka projeleri etkilenmez.
- **Proje adı:** Kod tarafında klasör hâlâ "UNIStainNet" olarak anılıyor; HPC tarafında her şeye `patholia_unistain` önekini koyduk. İkisi farklı: kaynak kodu adı = `UNIStainNet`, runtime artifactları = `patholia_unistain_*`.
