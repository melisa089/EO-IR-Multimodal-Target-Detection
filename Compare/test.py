import os
import gc
import copy
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.dataset import YOLODataset

# ==============================================================================
# 0. ÖZEL MİMARİ SINIFLARI (PyTorch .pt Dosyalarını Yüklemek İçin Zorunludur)
# ==============================================================================
class MidFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        in_c = channels * 2
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_c, in_c // 4, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(in_c // 4, in_c, bias=False),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, f_rgb, f_ir):
        x = torch.cat([f_rgb, f_ir], dim=1)
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        x = x * w
        return self.proj(x)


class BaseTwoStreamMidFusion(DetectionModel):
    def _resolve_fusion_layout(self):
        if hasattr(self, "fuse_indices") and hasattr(self, "backbone_end"):
            return self.fuse_indices, self.backbone_end

        backbone_end = len(self.rgb_backbone)
        p5_idx = backbone_end - 1

        skip_indices = []
        for layer in self.model:
            f = getattr(layer, "f", -1)
            if isinstance(f, list):
                for idx in f:
                    if 0 <= idx < backbone_end and idx not in skip_indices and idx != p5_idx:
                        skip_indices.append(idx)

        skip_indices = sorted(skip_indices)
        if len(skip_indices) >= 2:
            fuse_indices = (skip_indices[0], skip_indices[1], p5_idx)
        else:
            fuse_indices = (4, 6, p5_idx)

        self.fuse_indices = fuse_indices
        self.backbone_end = backbone_end
        return fuse_indices, backbone_end

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        fuse_indices, backbone_end = self._resolve_fusion_layout()
        idx_p3, idx_p4, idx_p5 = fuse_indices

        rgb_feats = {}
        h_rgb = x_rgb
        for idx, layer in enumerate(self.rgb_backbone):
            h_rgb = layer(h_rgb)
            if idx in fuse_indices:
                rgb_feats[idx] = h_rgb

        ir_feats = {}
        h_ir = x_ir
        for idx, layer in enumerate(self.ir_backbone):
            h_ir = layer(h_ir)
            if idx in fuse_indices:
                ir_feats[idx] = h_ir

        p3 = self.fuse_p3(rgb_feats[idx_p3], ir_feats[idx_p3])
        p4 = self.fuse_p4(rgb_feats[idx_p4], ir_feats[idx_p4])
        p5 = self.fuse_p5(rgb_feats[idx_p5], ir_feats[idx_p5])

        y = {idx_p3: p3, idx_p4: p4, idx_p5: p5}
        h = p5
        for idx, layer in enumerate(self.model, start=backbone_end):
            if layer.f != -1:
                h = y[layer.f] if isinstance(layer.f, int) else [h if j == -1 else y[j] for j in layer.f]
            h = layer(h)
            y[idx] = h

        return h


class TwoStreamYOLOv8MidFusion(BaseTwoStreamMidFusion): pass
class TwoStreamYOLO8MidFusion(BaseTwoStreamMidFusion): pass
class TwoStreamYOLO11MidFusion(BaseTwoStreamMidFusion): pass
class TwoStreamYOLOv11MidFusion(BaseTwoStreamMidFusion): pass
class TwoStreamYOLO12MidFusion(BaseTwoStreamMidFusion): pass
class TwoStreamYOLOv12MidFusion(BaseTwoStreamMidFusion): pass
class TwoStreamYOLO26MidFusion(BaseTwoStreamMidFusion): pass
class TwoStreamYOLOv26MidFusion(BaseTwoStreamMidFusion): pass


# ==============================================================================
# 1. KULLANICI AYARLARI
# ==============================================================================
DATASET_DIR = Path(r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Preprocessed_Fusion")
OUTPUT_DIR = Path(r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\Compare\test_official")

MODELS = {
    "YOLOv8s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv8s_Fusion\results\mid_fusion_yolov8s\weights\best.pt",
    "YOLO11s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv11s_Fusion\results\mid_fusion_yolo11s\weights\best.pt",
    "YOLO12s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv12s_Fusion\results\mid_fusion_yolo12s\weights\best.pt",
    "YOLO26s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv26s_Fusion\results\mid_fusion_yolo26s\weights\best.pt",
    # Tekil eğitim modellerini tamamladığında buraya ekleyebilirsin:
    # "YOLO26s-RGB":   r"C:\Users\melis\Desktop\BitirmeProjesi\weights\yolo26s_rgb.pt",
    # "YOLO26s-IR":    r"C:\Users\melis\Desktop\BitirmeProjesi\weights\yolo26s_ir.pt",
}

CLASS_NAMES = ['car', 'truck', 'bus', 'van', 'freight_car']
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. RESMİ LETTERBOX DESTEKLİ MULTIMODAL VALIDATOR
# ==============================================================================
class MultimodalYOLODataset(YOLODataset):
    def __init__(self, *args, modality="both", **kwargs):
        self.modality = modality
        super().__init__(*args, **kwargs)

    def img2label_paths(self, img_paths):
        return [
            str(Path(p).parent.parent / "labels" / f"{Path(p).stem}.txt")
            for p in img_paths
        ]

    def load_image(self, i, rect_mode=True):
        im_rgb, (h0, w0), (h, w) = super().load_image(i, rect_mode=rect_mode)
        rgb_path = Path(self.im_files[i])
        ir_path = rgb_path.parent.parent / "images_ir" / rgb_path.name
        im_ir = cv2.imread(str(ir_path))
        if im_ir is None:
            im_ir = im_rgb.copy()
        else:
            if (im_ir.shape[0], im_ir.shape[1]) != (im_rgb.shape[0], im_rgb.shape[1]):
                im_ir = cv2.resize(im_ir, (im_rgb.shape[1], im_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        if self.modality == "rgb_only":
            im_ir = np.zeros_like(im_ir)
        elif self.modality == "ir_only":
            im_rgb = np.zeros_like(im_rgb)

        im_6ch = np.concatenate([im_rgb, im_ir], axis=2)
        return im_6ch, (h0, w0), (h, w)


class MidFusionValidator(DetectionValidator):
    def __init__(self, *args, modality="both", **kwargs):
        self.modality = modality
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="val", batch=None):
        return MultimodalYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,
            hyp=self.args,
            rect=self.args.rect,
            cache=False,
            single_cls=self.args.single_cls,
            stride=int(self.stride),
            pad=0.5,
            prefix="",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=1.0,
            modality=self.modality
        )

    def warmup(self, imgsz=(1, 6, 640, 640)):
        super().warmup(imgsz=(1, 6, self.args.imgsz, self.args.imgsz))


# ==============================================================================
# 3. DEĞERLENDİRME MOTORU
# ==============================================================================
def evaluate_model(model_name, weight_path, yaml_path):
    print(f"\n{'='*60}")
    print(f"Resmi Validator ile Test Ediliyor: {model_name}")
    print(f"{'='*60}")

    if not os.path.exists(weight_path):
        print(f"[UYARI] Ağırlık dosyası bulunamadı, atlanıyor: {weight_path}")
        return None, None

    name_lower = model_name.lower()
    is_unimodal = ("rgb" in name_lower or "ir" in name_lower) and ("fusion" not in name_lower)

    ckpt = torch.load(weight_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu", weights_only=False)
    model = ckpt["model"].half().eval() if torch.cuda.is_available() else ckpt["model"].float().eval()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6

    val_args = dict(
        model=str(weight_path),
        data=str(yaml_path),
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        rect=True,
        half=torch.cuda.is_available(),
        conf=0.001,
        iou=0.70,   # Ultralytics resmi validation IoU eşiği
        project=str(OUTPUT_DIR),
        name=model_name,
        exist_ok=True,
        plots=True  # Karışıklık matrisi (Confusion Matrix) ve PR eğrilerini de üretir!
    )

    if is_unimodal:
        # Standart 3 kanallı tekil YOLO modeli için
        unimodal_yaml = OUTPUT_DIR / "data_unimodal.yaml"
        img_folder = "images_ir" if "ir" in name_lower else "images"
        u_cfg = {
            "path": str(DATASET_DIR),
            "train": f"train/{img_folder}",
            "val": f"val/{img_folder}",
            "test": f"test/{img_folder}",
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES
        }
        with open(unimodal_yaml, "w", encoding="utf-8") as f:
            yaml.dump(u_cfg, f, allow_unicode=True)

        val_args["data"] = str(unimodal_yaml)
        yolo_wrapper = YOLO(weight_path)
        res_obj = yolo_wrapper.val(**{k: v for k, v in val_args.items() if k != "model"})
        stats_dict = res_obj.results_dict
        speed_dict = res_obj.speed
        class_ap50_list = res_obj.box.ap50
    else:
        # 6 kanallı Mid-Fusion modelleri için
        validator = MidFusionValidator(args=val_args, modality="both")
        stats_dict = validator(model=model)
        speed_dict = validator.speed
        class_ap50_list = validator.metrics.box.ap50

    p_all = float(stats_dict.get("metrics/precision(B)", 0.0))
    r_all = float(stats_dict.get("metrics/recall(B)", 0.0))
    map50 = float(stats_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(stats_dict.get("metrics/mAP50-95(B)", 0.0))
    f1_all = 2 * (p_all * r_all) / (p_all + r_all + 1e-16)

    inf_ms = float(speed_dict.get("inference", 0.0))
    fps = 1000.0 / inf_ms if inf_ms > 0 else 0.0

    summary = {
        "Model": model_name,
        "mAP@50": round(map50, 4),
        "mAP@50:95": round(map50_95, 4),
        "Precision": round(p_all, 4),
        "Recall": round(r_all, 4),
        "F1-Score": round(f1_all, 4),
        "Latency (ms)": round(inf_ms, 2),
        "FPS": round(fps, 1),
        "Params (M)": round(total_params, 2)
    }

    class_ap50 = {"Model": model_name}
    for idx, cls_name in enumerate(CLASS_NAMES):
        val = float(class_ap50_list[idx]) if idx < len(class_ap50_list) else 0.0
        class_ap50[cls_name] = round(val, 4)

    del model, ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, class_ap50


# ==============================================================================
# 4. GRAFİKSEL KARŞILAŞTIRMA FONKSİYONLARI (300 DPI)
# ==============================================================================
def plot_comparisons(df_summary, df_classes, output_dir):
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    metrics_to_plot = ["mAP@50", "mAP@50:95", "Precision", "Recall", "F1-Score"]
    df_melted = df_summary.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metrik", value_name="Değer")

    barplot = sns.barplot(data=df_melted, x="Metrik", y="Değer", hue="Model", palette="Set2", ax=ax)
    ax.set_title("Modellerin Genel Tespit Performansı Karşılaştırması", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Skor (0 - 1)")
    ax.set_xlabel("")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    for container in barplot.containers:
        barplot.bar_label(container, fmt="%.3f", fontsize=8, padding=3, rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / "1_genel_metrik_karsilastirmasi.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    sns.scatterplot(
        data=df_summary, x="FPS", y="mAP@50", hue="Model",
        size="Params (M)", sizes=(150, 500), palette="deep", ax=ax, legend="brief"
    )
    for _, row in df_summary.iterrows():
        ax.annotate(
            f"{row['Model']}\n({row['mAP@50']:.3f} mAP | {row['FPS']} FPS)",
            (row["FPS"], row["mAP@50"]),
            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, fontweight="bold"
        )
    ax.set_title("Doğruluk (mAP@50) vs. Çıkarım Hızı (FPS) Analizi", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Çıkarım Hızı (FPS)")
    ax.set_ylabel("mAP@50")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "2_hiz_dogruluk_dengesi.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    df_cls_melted = df_classes.melt(id_vars="Model", value_vars=CLASS_NAMES, var_name="Sınıf", value_name="mAP@50")
    cls_plot = sns.barplot(data=df_cls_melted, x="Sınıf", y="mAP@50", hue="Model", palette="Paired", ax=ax)
    ax.set_title("Araç Sınıflarına Göre Model Başarım Karşılaştırması (mAP@50)", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mAP@50")
    ax.set_xlabel("Hedef Sınıflar")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    for container in cls_plot.containers:
        cls_plot.bar_label(container, fmt="%.2f", fontsize=8, padding=3, rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / "3_sinif_bazli_karsilastirma.png", dpi=300)
    plt.close()


# ==============================================================================
# 5. ANA ÇALIŞTIRMA BLOĞU
# ==============================================================================
if __name__ == "__main__":
    rgb_folder_name = "images" if (DATASET_DIR / "test" / "images").exists() else "images_rgb"
    yaml_path = OUTPUT_DIR / "data_fusion_test.yaml"
    cfg = {
        "path": str(DATASET_DIR),
        "train": f"train/{rgb_folder_name}",
        "val": f"val/{rgb_folder_name}",
        "test": f"test/{rgb_folder_name}",
        "channels": 6,
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    all_summaries = []
    all_class_results = []

    for model_name, weight_path in MODELS.items():
        summary, class_res = evaluate_model(model_name, weight_path, yaml_path)
        if summary is not None:
            all_summaries.append(summary)
            all_class_results.append(class_res)

    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        df_classes = pd.DataFrame(all_class_results)[["Model"] + CLASS_NAMES]

        print("\n" + "="*80)
        print("GENEL MODEL KARŞILAŞTIRMA TABLOSU (RESMİ LETTERBOX & VALIDATOR)")
        print("="*80)
        print(df_summary.to_string(index=False))

        print("\n" + "="*80)
        print("SINIF BAZLI mAP@50 KARŞILAŞTIRMA TABLOSU")
        print("="*80)
        print(df_classes.to_string(index=False))

        df_summary.to_csv(OUTPUT_DIR / "genel_sonuclar.csv", index=False)
        df_classes.to_csv(OUTPUT_DIR / "sinif_bazli_sonuclar.csv", index=False)

        plot_comparisons(df_summary, df_classes, OUTPUT_DIR)
        print(f"\nTüm tablolar, Confusion Matrix'ler ve 300 DPI grafikler kaydedildi:\n-> {OUTPUT_DIR}")