# Patholia — KCL HPC Eğitim Rehberi

> Linux'a yeni başlayanlara yönelik. Her adım sırayla, komutları kopyala-yapıştır.
> Tüm gerçek iş **bir script** ile başlıyor:
> - `setup.sh` → repo clone + ortam + paketler + veri seti + **batch1.sh'i otomatik kuyruğa atar**
> - `batch1.sh ... batch5.sh` → 5 ardışık eğitim job'u (zincir kendi kendine ilerler, 100 epoch)

---

## Genel Bakış

KCL CREATE HPC'de A30 GPU'lu node üzerinde Patholia/Phikon-v2-StainNet_2 modelini MIST veri setiyle (1024×1024) eğiteceğiz.

**Akış:**
1. HPC'ye bağlan
2. `setup.sh`'i HPC'ye al (curl ile GitHub'dan tek satır)
3. `setup.sh` job'unu gönder (≈1-2 saat) — clone + ortam + paketler + veri seti + batch1 otomatik kuyruğa atılır
4. Beklersin — 5'li zincir otomatik çalışır (100 epoch, her batch ≤48 saat)
5. Eğitim bitince checkpoint'leri PC'ne çek
6. HPC'yi temizle

**Senin yapacağın aktif iş:** ~10 dakika (komutları yapıştırmak).
**Bekleme:** Setup 1-2 saat, eğitim toplam 3-5 gün (otomatik, müdahale etmen gerekmez).

---

## Başlamadan Önce

- KCL CREATE hesap kullanıcı adı + şifre (yoksa: <https://docs.er.kcl.ac.uk/CREATE/requesting_access/>)
- MIST veri seti Drive klasörünün `Anyone with the link → Viewer` paylaşımı:
  <https://drive.google.com/drive/folders/146V99Zv1LzoHFYlXvSDhKmflIL-joo6p>
- Eğitim sonu checkpoint'leri (~2-5 GB) çekmek için PC'nde yer

---

# 🖥️ HPC'de Yapılacaklar

## Adım 1 — HPC'ye Bağlan

Tarayıcıda <https://portal.er.kcl.ac.uk/> aç, giriş yap → **Clusters → CREATE Shell**. Siyah pencerede şuna benzer satır görmelisin:
```
[k1234567@erc-hpc-login3 ~]$
```

> **Yapıştırma:** Komutu seç → kopyala → terminal penceresine sağ tıkla (veya `Ctrl + Shift + V`) → Enter.

## Adım 2 — setup.sh'i HPC'ye Al

setup.sh kendi içinde repo clone'u, ortam kurulumu ve batch1 submit'i yapıyor — yani tek dosya yeter. GitHub'dan direkt çek:

```bash
mkdir -p /scratch/prj/hpc_training/patholia/logs
cd ~
curl -O https://raw.githubusercontent.com/Patholia/Phikon-v2-StainNet_2/main/scripts/hpc/setup.sh
```

> **Farklı bir scratch yolu kullanmak istersen** önce `setup.sh`'i `nano setup.sh` ile aç, `WORK=` satırını değiştir.

## Adım 3 — Setup Job'unu Gönder (Eğitim Otomatik Başlar)

```bash
sbatch setup.sh
```

Çıktı: `Submitted batch job 12345678` → numarayı not al.

**Setup ne yapıyor (~1-2 saat):**
- Repo'yu klonlar (`$WORK/code/`)
- Conda env + PyTorch + paketler
- Phikon-v2 model ağırlıkları (cache)
- MIST dataset Drive'dan indir + doğru yapıda aç
- Dataset klasör yapısını doğrula
- **batch1.sh'i otomatik kuyruğa atar** — eğitim zinciri başlar

**Kontrol et:**
```bash
squeue -u $USER                              # setup + eğitim job'larını gör
tail -f /scratch/prj/hpc_training/patholia/logs/setup_<JOB_ID>.out
```

Setup bittiğinde log'un sonunda şunu görmelisin:
```
SETUP COMPLETE — eğitim zinciri başlatılıyor
[setup] batch1.sh submitted as job <BATCH1_JID>
```

Bu noktadan sonra eğitim zinciri kendiliğinden ilerler.

**Ne olur sonrasında?**

| Job | Hedef epoch | Süre | Sonraki |
|-----|-------------|------|---------|
| batch1 | 20 | ≤48 saat | batch2 (otomatik) |
| batch2 | 40 | ≤48 saat | batch3 (otomatik) |
| batch3 | 60 | ≤48 saat | batch4 (otomatik) |
| batch4 | 80 | ≤48 saat | batch5 (otomatik) |
| batch5 | 100 | ≤48 saat | — (son) |

Her batch script'i `last.ckpt` dosyasını okuyup PyTorch Lightning'in otomatik resume mantığıyla kaldığı epoch'tan devam eder. Bittiğinde bir sonraki batch'i `sbatch` ile tetikler. Sen müdahale etmiyorsun.

**Toplam:** 100 epoch, max ~5 gün. Çoğunlukla daha erken biter.

Bu noktadan sonra **bilgisayarını/terminalini kapatabilirsin** — SLURM job'ları arka planda çalışıyor.

---

## Adım 4 — İlerlemeyi İzleme (İSTEĞE BAĞLI)

İstediğin zaman HPC'ye dön ve durumu gör.

**Yeni oturuma girdiysen önce $WORK'u set et:**
```bash
export WORK=/scratch/prj/hpc_training/patholia/patholia_unistain
cd $WORK
```

**Kuyruktaki job'lar:**
```bash
squeue -u $USER
```

**Şu an çalışan job'un canlı log'u** (JOB_ID = `R` durumundaki):
```bash
tail -f patholia_b*_<JOB_ID>.out
```
Çıkış: `Ctrl + C` (job durmaz).

**Tamamlanan epoch sayısı:**
```bash
ls -1 $WORK/checkpoints/mist_1024/mist_1024_epoch=*.ckpt 2>/dev/null | tail -3
```
Dosya isimlerindeki `epoch=NNN` en büyüğü gerçek epoch sayısı. 99'a (yani 100. epoch) ulaştığında zincir bitmiş.

**Belirli bir job'u iptal et:**
```bash
scancel <JOB_ID>
```

**Tüm kuyruğu iptal et:**
```bash
scancel -u $USER
```

> **OOM hatası — iki tür var, karıştırma:**
>
> **1) Sistem RAM OOM** — `.err`'de `oom_kill event` / `Killed` görürsün (Python traceback YOK). Job'a yeterli RAM/CPU verilmemiştir. Batch dosyalarında `#SBATCH --mem=64G` ve `#SBATCH --cpus-per-task=8` olmalı (bu repo'da zaten var). Eğer hâlâ olursa `--mem`'i 96G veya 128G yap.
>
> **2) GPU OOM** — `.err`'de `CUDA out of memory` + Python traceback görürsün. GPU belleği (A30 = 24GB) yetmiyor. batch_size'ı düşür:
> ```bash
> sed -i 's/--batch_size 4/--batch_size 2/' $WORK/code/scripts/hpc/batch*.sh
> ```
>
> Her iki durumda da düzelttikten sonra `last.ckpt` scratch'te olduğu için kaldığı yerden tekrar başlatabilirsin:
> ```bash
> sbatch --export=ALL,WORK=$WORK $WORK/code/scripts/hpc/batch1.sh
> ```

---

# 🖥️ Kendi PC'nde — Checkpoint'leri İndir

Eğitim bittiğinde (`squeue -u $USER` boş, son log'da "BATCH 5 (FINAL) tamam" yazıyor) modeli HPC'den PC'ne çek.

## Adım 5 — PC'ye Pull

Kendi bilgisayarında **bir terminal** aç (Windows: PowerShell, Mac: Terminal, Linux: Terminal). PC'de bir indirme klasörü oluştur, sonra:

### Seçenek 1 — scp (en basit, her platformda çalışır)

```bash
scp -r <KULLANICI_ADIN>@create.kcl.ac.uk:/scratch/prj/hpc_training/patholia/patholia_unistain/checkpoints/mist_1024 ./patholia_ckpts
```

`<KULLANICI_ADIN>` yerine KCL kullanıcı adını yaz (örn. `k1234567`). KCL şifren + MFA kodun sorulur. 5-30 dakikada indirir.

### Seçenek 2 — rsync (önerilen — yarıda kalsa devam eder)

```bash
rsync -avh --progress <KULLANICI_ADIN>@create.kcl.ac.uk:/scratch/prj/hpc_training/patholia/patholia_unistain/checkpoints/mist_1024/ ./patholia_ckpts/
```

> Windows'ta rsync yoksa WSL kur ya da Seçenek 1 ile yetin.

### Seçenek 3 — WinSCP (Windows GUI)

<https://winscp.net/> → kur → Host: `create.kcl.ac.uk`, kullanıcı adı + şifre ile bağlan → sağdaki tarayıcıdan `/scratch/prj/hpc_training/patholia/patholia_unistain/checkpoints/mist_1024/` yoluna git → tüm `.ckpt` dosyalarını sürükleyip soldaki PC klasörüne bırak.

### Hangi dosyalar?

`mist_1024/` klasöründe ~4-5 dosya olmalı:
- `last.ckpt` — en son state (mutlaka çek)
- `mist_1024_epoch=NNN_step=NNNNNN.ckpt` × 3 — en iyi val/lpips skorlu top-3 model

Toplam ~2-5 GB.

---

## Adım 6 — HPC'yi Temizle (ZORUNLU)

**Önce PC'de indirdiğin dosyaları gör** (`ls patholia_ckpts/`), sonra HPC terminalinde:

```bash
scancel -u $USER
conda deactivate 2>/dev/null

rm -rf $WORK

rm -rf $HOME/.cache/huggingface $HOME/.cache/torch $HOME/.cache/pip $HOME/.cache/wandb 2>/dev/null
rm -f $HOME/patholia_*.out $HOME/patholia_*.err 2>/dev/null

find $HOME -maxdepth 4 -name "*patholia*" 2>/dev/null
find /scratch -path "*patholia*" 2>/dev/null
echo "TEMİZ"
```

`TEMİZ` tek başına çıkıyorsa (öncesinde liste yoksa) tüm izler silinmiş.

---

# Hatırlatma Kartı — Her Yeni HPC Oturumunda

```bash
export WORK=/scratch/prj/hpc_training/patholia/patholia_unistain
export HF_HOME=$WORK/.hf_cache_patholia
module load anaconda3/2022.10-gcc-13.2.0
module load cuda/12.2.1-gcc-13.2.0
source activate $WORK/.conda_envs/patholia_unistain
cd $WORK/code
```

---

# Sıkça Karşılaşılan Sorunlar

| Sorun | Anlam | Çözüm |
|-------|-------|-------|
| `module: command not found` | Yeni oturum | `module load anaconda3/2022.10-gcc-13.2.0` |
| `FileNotFoundError: ...trainA` | Dataset extraction çift-nesting | `rm -rf $WORK/MIST` ve setup'ı tekrar çalıştır |
| `oom_kill event` / `Killed` (traceback yok) | Sistem RAM yetersiz | Batch dosyalarında `--mem` artır (zaten 64G; gerekiyorsa 96G/128G) |
| `CUDA out of memory` (traceback var) | GPU belleği yetersiz | `sed` ile batch_size 2 yap, batch1'i tekrar gönder |
| `torch.cuda.is_available() = False` | CUDA modülü eksik | `module load cuda/12.2.1-gcc-13.2.0` |
| `gdown: permission denied` | Drive paylaşımı kapalı | Veri seti klasörünü "Anyone with the link" yap |
| Job sürekli `PD` | GPU kuyruğu dolu | Bekle |
| `scp: connection refused` | SSH bağlantı sorunu | KCL VPN/MFA + kullanıcı adı doğru mu? |
| `Disk quota exceeded` | Home alanı doldu | Adım 6'daki cache temizliği |

---

# Notlar

- **Zincirleme tamamen otomatik:** `batch1.sh` submit ettikten sonra ne SSH oturumun ne de bilgisayarın açık olması gerekir. SLURM scheduler zinciri yönetir.
- **Auto-resume:** Her batch script'i `last.ckpt`'i okur. Time-limit yedi mi, hata mı verdi fark etmez — bir sonraki batch ya bu zincir ya manuel resubmit ile kaldığı epoch'tan devam eder.
- **PyTorch Lightning resume mantığı:** `--max_epochs 40` ile resume yaptığında, ckpt'de 20 epoch yapıldıysa sadece kalan 20'yi (epoch 20-39) eğitir. Her batch'in `--max_epochs`'u kümülatif hedef olarak yazılı (20, 40, 60, 80, 100).
- **Drive yok:** Setup script Drive'dan SADECE veri setini indirir. Eğitim sonuçları HPC scratch'te kalır, sen PC'ne çekersin (Adım 5).
- **Scratch policy:** KCL CREATE bazı durumlarda 30-60 gün dokunulmayan scratch dosyalarını siler. Eğitim biter bitmez PC'ne çek + Adım 6'yı çalıştır.
