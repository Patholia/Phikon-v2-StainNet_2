"""
pannuke_to_ihc.py
=================
HuggingFace RationAI/PanNuke -> UNIStainNet -> IHC NPY

Pipeline (her görüntü için):
    HF PanNuke Breast 256x256 (40x büyütme)
        -> downsample 2x -> 128x128  (20x büyütme, MIST ölçeği)
        -> beyaz pad 448px her kenara -> 1024x1024 (UNIStainNet giriş boyutu)
        -> UNIStainNet inference (ER / PR / HER2 / Ki67)
        -> center crop -> 128x128 (pad'ı çıkar)
        -> upsample 2x -> 256x256 (40x'e geri, orijinal PanNuke boyutu)
        -> NPY olarak kaydet (HoverNext-IHC formatı)

Inference logic: scripts/eval/eval_mist_1024.py'deki doğrulanmış pattern.
    - Phikon-v2 direkt transformers.AutoModel ile yükleniyor
    - 4×4 grid sub-crops × 14×14 patches = 56×56 → adaptive pool → 32×32
    - model.generate(he, uni_features, labels, guidance_scale=1.0)

Çıktı yapısı (HoverNext-IHC data_utils.py'ın düz NPY formatı):
    <output_dir>/
    ├── ER/
    │   ├── images.npy   [N, 256, 256, 3] uint8
    │   └── labels.npy   [N, 256, 256, 6] float32   (PanNuke 6-channel format)
    ├── PR/   ...
    ├── HER2/ ...
    └── Ki67/ ...

Kullanım:
    cd UNIStainNet-main
    python veri_boyut/pannuke_to_ihc.py \
        --output_dir  /path/to/ihc_output \
        --checkpoint  /path/to/unistainnet.ckpt \
        --markers     ER PR HER2 Ki67 \
        --device      cuda

Gereksinimler:
    pip install datasets numpy opencv-python pillow tqdm torch transformers
"""

import argparse
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ── Sabitler ──────────────────────────────────────────────────────────────
SRC_SIZE   = 256      # PanNuke orijinal boyut (40x büyütme)
MID_SIZE   = 128      # MIST ölçeği (20x büyütme)
MODEL_SIZE = 1024     # UNIStainNet inference boyutu
PAD        = (MODEL_SIZE - MID_SIZE) // 2   # = 448 piksel beyaz pad her kenara
MARKERS    = ["ER", "PR", "HER2", "Ki67"]
MARKER_TO_STAIN_ID = {"HER2": 0, "Ki67": 1, "ER": 2, "PR": 3}

# UNI / Phikon-v2 normalization (ImageNet stats)
UNI_NORM_MEAN = [0.485, 0.456, 0.406]
UNI_NORM_STD  = [0.229, 0.224, 0.225]


# ── HuggingFace PanNuke Yükleyici ──────────────────────────────────────────

def load_pannuke_breast_hf(cache_dir=None, breast_cache=None):
    """HuggingFace RationAI/PanNuke -> sadece Breast döndür (streaming mode).

    Streaming kullandığımız için HF parquet'leri geçici olarak indirilir
    ama Drive'a sadece breast filtresinden geçenler kaydedilir.
    Drive disk kullanımı: ~3-5 GB (full cache) yerine ~300-500 MB (breast only).

    Args:
        cache_dir   : Streaming geçici cache (kullanılmaz ama HF için lazım)
        breast_cache: Filtrelenmiş breast verisinin kalıcı kaydedileceği .npz yolu.
                      Eğer dosya varsa yeniden indirme yapılmaz, direkt yüklenir.

    PanNuke 5 sınıf: 0=neoplastic, 1=inflammatory, 2=connective,
                    3=dead, 4=epithelial
    HF format: image (PIL), instances (list[PIL bin mask]), categories (list[int]),
               tissue (str)
    """
    # 1) Önceden filtrelenmiş breast cache varsa direkt yükle
    if breast_cache and os.path.exists(breast_cache):
        print(f"Breast cache bulundu, yükleniyor: {breast_cache}")
        data = np.load(breast_cache)
        images = data["images"]
        labels = data["labels"]
        print(f"  {len(images)} Breast patch yüklendi (yeniden indirme yok)")
        return images, labels

    # 2) Yoksa streaming ile indir + filtrele + kaydet
    from datasets import load_dataset

    print("HuggingFace RationAI/PanNuke streaming başlıyor (sadece Breast filtreleniyor)...")

    all_images, all_labels = [], []

    for split in ["fold1", "fold2", "fold3"]:
        try:
            fold_data = load_dataset(
                "RationAI/PanNuke", split=split,
                streaming=True, cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"  {split} atlandı: {e}")
            continue

        breast_count = 0
        seen = 0
        for sample in tqdm(fold_data, desc=f"{split} stream"):
            seen += 1
            if sample["tissue"] != "Breast":
                continue
            breast_count += 1
            img = np.array(sample["image"].convert("RGB"))  # [256, 256, 3]
            label = hf_to_pannuke_mask(
                sample["instances"], sample["categories"], size=SRC_SIZE,
            )
            all_images.append(img)
            all_labels.append(label)
        print(f"  {split}: {seen} patch tarandı, {breast_count} Breast bulundu")

    if not all_images:
        raise RuntimeError("Hiç Breast patch bulunamadı — PanNuke stream başarısız.")

    images = np.stack(all_images).astype(np.uint8)
    labels = np.stack(all_labels).astype(np.float32)
    print(f"Toplam: {len(images)} Breast patch")

    # 3) Breast-only cache'i Drive'a kaydet (diğer notebook'lar için)
    if breast_cache:
        os.makedirs(os.path.dirname(breast_cache), exist_ok=True)
        print(f"Breast cache Drive'a kaydediliyor: {breast_cache}")
        np.savez_compressed(breast_cache, images=images, labels=labels)
        size_mb = os.path.getsize(breast_cache) / (1024 ** 2)
        print(f"  Kaydedildi ({size_mb:.1f} MB)")

    return images, labels


def hf_to_pannuke_mask(instances, categories, size=256):
    """HF instance listesini PanNuke 6-kanal mask formatına çevir.

    Format [H, W, 6]:
        kanal 0 : instance map (her hücreye benzersiz int ID, 0=bg)
        kanal 1 : sınıf haritası (1-5 PanNuke sınıfları, 0=bg)
        kanal 2-5 : per-class instance binary mask (cat 0-3'ün her biri ayrı kanal)

    Not: Sınıf indeksleri convert_pannuke_to_conic.py'deki (1-5 başlangıçlı)
    konvansiyonla uyumlu — class_map değeri = cat + 1 → background = 0, sınıflar 1-5.
    Bu sayede `_load_marker_data` ve `add_3c_gt_fast` doğru sınıf etiketlerini görür.
    """
    mask = np.zeros((size, size, 6), dtype=np.float32)
    inst_map  = mask[..., 0]
    class_map = mask[..., 1]

    for inst_id, (inst_img, cat) in enumerate(zip(instances, categories), start=1):
        inst_arr = np.array(inst_img.convert("L")) > 0
        inst_arr = inst_arr.astype(np.float32)
        if inst_arr.shape[0] != size or inst_arr.shape[1] != size:
            inst_arr = cv2.resize(inst_arr, (size, size),
                                  interpolation=cv2.INTER_NEAREST)
            inst_arr = (inst_arr > 0).astype(np.float32)

        # Instance map: benzersiz ID
        inst_map[inst_arr > 0] = inst_id
        # Class map: PanNuke konvansiyonu (1-5, 0=bg) — HF 0-4 → +1
        class_map[inst_arr > 0] = cat + 1
        # Per-class binary kanal (sadece 4 ana sınıf için ayrı kanal,
        # epithelial = cat=4 kanal değil, sadece class_map'te)
        if cat < 4:
            mask[..., cat + 2][inst_arr > 0] = inst_id
    return mask


# ── Görüntü Dönüşümleri (boyut + zoom oranı korunarak) ────────────────────

def downsample_40x_to_20x(img_256):
    """40x büyütmeli 256 PanNuke → 20x büyütmeli 128 (MIST ölçeği).

    cv2.INTER_AREA: downsampling için optimal (alan ortalamalı).
    Zoom oranı 40x → 20x kayboluyor: piksel başına dokunun fiziksel
    boyutu 2x büyüyor. MIST eğitimi 20x büyütmeyle yapıldı.
    """
    return cv2.resize(img_256, (MID_SIZE, MID_SIZE), interpolation=cv2.INTER_AREA)

def pad_white_to_1024(img_128):
    """128×128 → 1024×1024 her kenara 448px beyaz dolgu.

    UNIStainNet 1024×1024 girişe alıştırılmıştır. Daha küçük bir dokuyu
    1024'e direkt resize etmek zoom oranını bozar; bunun yerine merkeze
    yerleştirip kenarları beyaz dokuyla dolduruyoruz. Beyaz = WSI'nin boş
    cam bölümü gibi davranır, model bu kısımları tahmin etmez.
    """
    return cv2.copyMakeBorder(
        img_128, PAD, PAD, PAD, PAD,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )

def center_crop_128(img_1024):
    """1024×1024 → 128×128 (pad'ı çıkar, sadece gerçek doku verisi)."""
    return img_1024[PAD: PAD + MID_SIZE, PAD: PAD + MID_SIZE]

def upsample_20x_to_40x(img_128):
    """20x büyütmeli 128 → 40x büyütmeli 256 (orijinal PanNuke boyutu).

    cv2.INTER_CUBIC: upsampling için keskin kenarları korur.
    """
    return cv2.resize(img_128, (SRC_SIZE, SRC_SIZE), interpolation=cv2.INTER_CUBIC)


# ── Phikon-v2 Feature Extraction (eval_mist_1024.py'den) ──────────────────

def load_phikon_v2(device):
    """Phikon-v2 modeli (UNI'nin açık erişimli muadili) yükle."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract_uni_features_1024(uni_model, he_01, device, spatial_pool_size=32):
    """1024×1024 H&E [0,1] → UNI/Phikon-v2 features [B, 1024, 1024].

    eval_mist_1024.py'deki extract_features_for_crop ile birebir aynı.
    4×4 grid sub-crops × 14×14 ViT patches (16'lık patch, 224×224 sub-crop) →
    56×56 grid → adaptive pool → 32×32 = 1024 token.
    """
    uni_transform = T.Normalize(mean=UNI_NORM_MEAN, std=UNI_NORM_STD)

    B, _, H, W = he_01.shape
    num_crops = 4
    patches_per_side = 14   # 224 / 16

    sub_crops = []
    crop_h = H // num_crops
    crop_w = W // num_crops
    for i in range(num_crops):
        for j in range(num_crops):
            sub = he_01[:, :, i*crop_h:(i+1)*crop_h, j*crop_w:(j+1)*crop_w]
            sub = F.interpolate(sub, size=(224, 224), mode='bicubic', align_corners=False)
            sub = torch.stack([uni_transform(s) for s in sub])
            sub_crops.append(sub)

    all_crops = torch.stack(sub_crops, dim=1).reshape(B * 16, 3, 224, 224).to(device)
    all_feats = uni_model(pixel_values=all_crops).last_hidden_state
    patch_tokens = all_feats[:, 1:, :]   # CLS hariç

    patch_tokens = patch_tokens.reshape(
        B, num_crops, num_crops, patches_per_side, patches_per_side, 1024
    )
    full_size = num_crops * patches_per_side   # 56
    full_grid = patch_tokens.permute(0, 1, 3, 2, 4, 5).reshape(
        B, full_size, full_size, 1024
    )

    if spatial_pool_size < full_size:
        grid_bchw = full_grid.permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(grid_bchw, spatial_pool_size)
        result = pooled.permute(0, 2, 3, 1)
    else:
        result = full_grid

    S = result.shape[1]
    return result.reshape(B, S * S, 1024)


# ── UNIStainNet Inference ─────────────────────────────────────────────────

def load_unistainnet(checkpoint_path, device):
    """UNIStainNetTrainer'ı checkpoint'ten yükle."""
    try:
        from src.models.trainer import UNIStainNetTrainer
    except ImportError as e:
        raise ImportError(
            f"UNIStainNet import hatası: {e}\n"
            "Bu scripti UNIStainNet-main klasöründen çalıştır:\n"
            "    cd UNIStainNet-main\n"
            "    python veri_boyut/pannuke_to_ihc.py ..."
        )

    model = UNIStainNetTrainer.load_from_checkpoint(
        checkpoint_path, map_location=device, strict=False
    )
    model = model.to(device).eval()
    print(f"UNIStainNet yüklendi: {checkpoint_path}")
    return model


@torch.no_grad()
def translate_patch(unistain_model, uni_model, img_np_1024, marker, device):
    """Tek bir 1024×1024 padlı görüntüyü tek bir IHC marker'a çevir.

    eval_mist_1024.py'deki generate_for_stain pattern'iyle birebir.

    Args:
        unistain_model : UNIStainNetTrainer
        uni_model      : Phikon-v2 (transformers AutoModel)
        img_np_1024    : np.ndarray [1024, 1024, 3] uint8 — beyaz padlı görüntü
        marker         : "ER" / "PR" / "HER2" / "Ki67"

    Returns:
        ihc_np : np.ndarray [1024, 1024, 3] uint8
    """
    # 1) H&E tensor → [-1, 1] (eğitimle aynı aralık)
    he = torch.from_numpy(img_np_1024).float().permute(2, 0, 1) / 255.0   # [3, 1024, 1024] in [0,1]
    he = (he - 0.5) / 0.5    # [-1, 1]
    he = he.unsqueeze(0).to(device)   # [1, 3, 1024, 1024]

    # 2) [0,1] versiyonu UNI feature extraction için
    he_01 = ((he + 1) / 2).clamp(0, 1)

    # 3) Phikon-v2 features
    uni_features = extract_uni_features_1024(uni_model, he_01, device)   # [1, 1024, 1024]

    # 4) Stain label
    labels = torch.tensor([MARKER_TO_STAIN_ID[marker]], device=device, dtype=torch.long)

    # 5) Generate (model.generate EMA generator kullanır)
    out = unistain_model.generate(he, uni_features, labels, guidance_scale=1.0)
    # out: [1, 3, 1024, 1024] in [-1, 1]

    # 6) [-1, 1] → [0, 255] uint8
    out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = ((out_np + 1.0) / 2.0) * 255.0
    return np.clip(out_np, 0, 255).astype(np.uint8)


# ── Ana Pipeline ───────────────────────────────────────────────────────────

def run(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device}")

    unistain_model = load_unistainnet(args.checkpoint, device)
    print("Phikon-v2 yükleniyor...")
    uni_model = load_phikon_v2(device)

    images, labels = load_pannuke_breast_hf(
        cache_dir=args.cache_dir,
        breast_cache=args.breast_cache,
    )

    for marker in args.markers:
        print(f"\n{'='*50}  {marker}  {'='*50}")
        out_dir = os.path.join(args.output_dir, marker)
        os.makedirs(out_dir, exist_ok=True)

        images_npy = os.path.join(out_dir, "images.npy")
        labels_npy = os.path.join(out_dir, "labels.npy")

        # Resume: var olan partial output'u yükle, kaldığı yerden devam et
        partial_npy = os.path.join(out_dir, f"_partial_{marker}.npy")
        if os.path.exists(partial_npy):
            partial = np.load(partial_npy)
            start_idx = len(partial)
            ihc_images = [partial[i] for i in range(start_idx)]
            del partial
            print(f"  RESUME: {start_idx}/{len(images)} patch zaten yapılmış, devam ediyor...")
        else:
            ihc_images = []
            start_idx = 0

        # Checkpoint sıklığı (saniyede patch'i sayıp ona göre flush)
        CHECKPOINT_EVERY = args.checkpoint_every

        pbar = tqdm(range(start_idx, len(images)), desc=marker, initial=start_idx, total=len(images))
        for idx in pbar:
            img = images[idx]
            img_128  = downsample_40x_to_20x(img)        # 256@40x → 128@20x
            img_1024 = pad_white_to_1024(img_128)         # 128 → 1024 (beyaz pad)
            ihc_1024 = translate_patch(                    # UNIStainNet inference
                unistain_model, uni_model, img_1024, marker, device
            )
            ihc_128  = center_crop_128(ihc_1024)           # 1024 → 128 (pad çıkar)
            ihc_256  = upsample_20x_to_40x(ihc_128)        # 128@20x → 256@40x
            ihc_images.append(ihc_256)

            # Periyodik partial save — disconnect olsa kaldığı yerden devam edebilir
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                np.save(partial_npy, np.stack(ihc_images).astype(np.uint8))
                pbar.set_postfix({"checkpoint": f"{idx+1}/{len(images)}"})

        # Final save
        np.save(images_npy, np.stack(ihc_images).astype(np.uint8))
        np.save(labels_npy, labels)
        print(f"  Kaydedildi: {images_npy}  ({len(ihc_images)} patch)")

        # Partial'i sil (artık final var)
        if os.path.exists(partial_npy):
            os.remove(partial_npy)

    print("\nTamamlandı.")
    print(f"Çıktı: {args.output_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HF PanNuke Breast → UNIStainNet → IHC NPY"
    )
    parser.add_argument("--output_dir", required=True,
        help="IHC çıktı klasörü — ER/, PR/, HER2/, Ki67/ burada oluşturulur")
    parser.add_argument("--checkpoint", required=True,
        help="UNIStainNet checkpoint (.ckpt) dosya yolu")
    parser.add_argument("--markers", nargs="+", default=MARKERS, choices=MARKERS,
        help=f"İşlenecek markerlar (varsayılan: {MARKERS})")
    parser.add_argument("--device", default="cuda", help="'cuda' veya 'cpu'")
    parser.add_argument("--cache_dir", default=None,
        help="HuggingFace dataset geçici cache (streaming için)")
    parser.add_argument("--breast_cache", default=None,
        help="Breast-only kalıcı cache .npz yolu (Drive'a kaydet — diğer marker "
             "notebook'ları yeniden indirme yapmaz). Önerilen: "
             "<DRIVE>/pannuke_raw/pannuke_breast.npz")
    parser.add_argument("--checkpoint_every", type=int, default=200,
        help="Her N patch'te partial output kaydet (disconnect güvenliği). "
             "Default 200 (~5-10 dk'da bir).")
    args = parser.parse_args()
    run(args)
