import gc
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

# ==============================================================================
# 1. VERİ SETİ VE ÇALIŞMA ORTAMI AYARLARI (Füzyon eğitimleriyle birebir aynı)
# ==============================================================================
DATASET_DIR = Path(r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Preprocessed_Fusion")
WORK_DIR = Path(__file__).resolve().parent
YAML_PATH = WORK_DIR / "baseline_rgb_yolo26s.yaml"
SAVE_PROJECT_DIR = WORK_DIR / "results"
RUN_NAME = "baseline_rgb_yolo26s"
MODEL_WEIGHTS = "yolo26s.pt"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU bulunamadı!")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Veri seti klasörü bulunamadı: {DATASET_DIR}")

    torch.cuda.empty_cache()

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Kullanılan GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"Veri Seti Yolu: {DATASET_DIR}")
    print(f"Sonuç Klasörü: {SAVE_PROJECT_DIR / RUN_NAME}")

    # RGB klasör adını otomatik kontrol et
    rgb_folder_name = "images" if (DATASET_DIR / "train" / "images").exists() else "images_rgb"

    # Standart 3 kanallı RGB YAML yapılandırması
    cfg = {
        "path": str(DATASET_DIR),
        "train": f"train/{rgb_folder_name}",
        "val": f"val/{rgb_folder_name}",
        "test": f"test/{rgb_folder_name}",
        "nc": 5,
        "names": ["car", "truck", "bus", "van", "freight car"]
    }
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # Standart 3 kanallı YOLO26s modelini yükle
    model = YOLO(MODEL_WEIGHTS)

    # Füzyon modelleri ile ADİL KARŞILAŞTIRMA için birebir aynı hiperparametreler
    train_args = dict(
        data=str(YAML_PATH),
        project=str(SAVE_PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        device=0,
        workers=2,
        cache=False,
        epochs=100,
        imgsz=640,
        batch=4,
        nbs=16,
        amp=True,
        seed=42,
        optimizer="SGD",
        lr0=0.008,
        cos_lr=True,
        close_mosaic=15,
        mosaic=1.0,
        copy_paste=0.0,
        erasing=0.0,
        scale=0.5,
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        iou=0.55
    )

    # Eğitimi başlat
    model.train(**train_args)

    # Bellek temizliği
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================================================
    # 2. EĞİTİM SONRASI OTOMATİK TEST SETİ DEĞERLENDİRMESİ
    # ==========================================================================
    print("\n--- YOLO26s SADECE RGB (UNIMODAL BASELINE) TEST SETİ SONUÇLARI ---")
    best_weights = SAVE_PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    best_model = YOLO(str(best_weights))

    total_params = sum(p.numel() for p in best_model.model.parameters()) / 1e6
    print(f"Toplam Parametre: {total_params:.2f} M")

    metrics = best_model.val(
        data=str(YAML_PATH),
        split="test",
        imgsz=640,
        batch=8,
        workers=0,
        rect=True,
        iou=0.70,
        project=str(SAVE_PROJECT_DIR),
        name=f"{RUN_NAME}_test_eval",
        exist_ok=True,
        plots=True
    )

    res = metrics.results_dict
    prec = res.get("metrics/precision(B)", 0.0)
    rec = res.get("metrics/recall(B)", 0.0)
    map50 = res.get("metrics/mAP50(B)", 0.0)
    map50_95 = res.get("metrics/mAP50-95(B)", 0.0)

    print("\n================ YOLO26s RGB BASELINE ÖZET ================")
    print(f"Sadece RGB (Unimodal) -> P: {prec:.4f} | R: {rec:.4f} | mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map50_95:.4f}")


if __name__ == "__main__":
    main()