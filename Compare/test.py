import os
import gc
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
import ultralytics.data.utils as data_utils
import ultralytics.data.dataset as data_dataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.dataset import YOLODataset

# ==============================================================================
# 0. images_ir ETİKET EŞLEŞTİRME PATCH'İ (Tekil IR Testi İçin Zorunlu)
# ==============================================================================
_orig_img2label_paths = data_utils.img2label_paths


def custom_img2label_paths(img_paths):
    fixed_paths = [
        str(Path(p).parent.parent / "images" / Path(p).name)
        if Path(p).parent.name == "images_ir"
        else p
        for p in img_paths
    ]
    return _orig_img2label_paths(fixed_paths)


data_utils.img2label_paths = custom_img2label_paths
data_dataset.img2label_paths = custom_img2label_paths


# ==============================================================================
# 1. ÖZEL MİMARİ SINIFLARI (PyTorch .pt Dosyalarını Yüklemek İçin Zorunlu)
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
# 2. KULLANICI AYARLARI VE MODEL YOLLARI
# ==============================================================================
DATASET_DIR = Path(r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Preprocessed_Fusion")
OUTPUT_DIR = Path(r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\Compare\test_final_ablation")

MODELS = {
    "YOLO26s-RGB":     r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv26s_Baseline\results\baseline_rgb_yolo26s\weights\best.pt",
    "YOLO26s-IR":      r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv26s_Baseline\results\baseline_ir_yolo26s\weights\best.pt",
    "YOLOv8s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv8s_Fusion\results\mid_fusion_yolov8s\weights\best.pt",
    "YOLO11s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv11s_Fusion\results\mid_fusion_yolo11s\weights\best.pt",
    "YOLO12s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv12s_Fusion\results\mid_fusion_yolo12s\weights\best.pt",
    "YOLO26s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv26s_Fusion\results\mid_fusion_yolo26s\weights\best.pt",
}

CLASS_NAMES = ['car', 'truck', 'bus', 'van', 'freight_car']
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 3. RESMİ LETTERBOX DESTEKLİ MULTIMODAL VALIDATOR
# ==============================================================================
class MultimodalYOLODataset(YOLODataset):
    def __init__(self, *args, modality="both", **kwargs):
        self.modality = modality
        super().__init__(*args, **kwargs)

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
# 4. DEĞERLENDİRME VE METRİK TOPLAMA MOTORU
# ==============================================================================
def evaluate_model(model_name, weight_path, yaml_path):
    print(f"\n{'='*65}")
    print(f"Değerlendiriliyor: {model_name}")
    print(f"{'='*65}")

    if not os.path.exists(weight_path):
        print(f"[UYARI] Ağırlık dosyası bulunamadı, atlanıyor: {weight_path}")
        return None, None, None, None

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
        iou=0.70,
        project=str(OUTPUT_DIR),
        name=model_name,
        exist_ok=True,
        plots=True  # BoxPR_curve (AUC), BoxF1_curve, BoxP_curve, BoxR_curve ve confusion_matrix üretir
    )

    if is_unimodal:
        unimodal_yaml = OUTPUT_DIR / f"data_{model_name.lower()}.yaml"
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
        metrics_obj = res_obj
        conf_matrix = res_obj.confusion_matrix.matrix.copy() if getattr(res_obj, "confusion_matrix", None) is not None else None
    else:
        validator = MidFusionValidator(args=val_args, modality="both")
        stats_dict = validator(model=model)
        speed_dict = validator.speed
        metrics_obj = validator.metrics
        conf_matrix = validator.confusion_matrix.matrix.copy() if getattr(validator, "confusion_matrix", None) is not None else None

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

    # Sınıf bazlı detaylı metrikler (mAP@50, Precision, Recall, F1)
    class_ap50_row = {"Model": model_name}
    detailed_class_rows = []
    for idx, cls_name in enumerate(CLASS_NAMES):
        try:
            p_c, r_c, ap50_c, ap_c = metrics_obj.box.class_result(idx)
        except Exception:
            p_c, r_c, ap50_c, ap_c = 0.0, 0.0, 0.0, 0.0
        f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-16)

        class_ap50_row[cls_name] = round(float(ap50_c), 4)
        detailed_class_rows.append({
            "Model": model_name,
            "Sınıf": cls_name,
            "Precision": round(float(p_c), 4),
            "Recall": round(float(r_c), 4),
            "F1-Score": round(float(f1_c), 4),
            "mAP@50": round(float(ap50_c), 4),
            "mAP@50:95": round(float(ap_c), 4),
        })

    del model, ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary, class_ap50_row, detailed_class_rows, conf_matrix


# ==============================================================================
# 5. TEZ KALİTESİNDE (300 DPI) KARŞILAŞTIRMA GRAFİKLERİ
# ==============================================================================
def plot_all_comparisons(df_summary, df_classes, conf_matrices, output_dir):
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # --- GRAFİK 1: Genel Tespit Metrikleri (Ablation + Mimari Kıyas) ---
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    metrics_to_plot = ["mAP@50", "mAP@50:95", "Precision", "Recall", "F1-Score"]
    df_melted = df_summary.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metrik", value_name="Değer")

    barplot = sns.barplot(data=df_melted, x="Metrik", y="Değer", hue="Model", palette="Set2", ax=ax)
    ax.set_title("Tekil Modalite (RGB/IR) ve Mid-Fusion Modellerinin Genel Performans Karşılaştırması", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Skor (0 - 1)")
    ax.set_xlabel("")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")

    for container in barplot.containers:
        barplot.bar_label(container, fmt="%.3f", fontsize=7, padding=3, rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / "1_genel_metrik_karsilastirmasi.png", dpi=300)
    plt.close()

    # --- GRAFİK 2: Hız (FPS) vs Doğruluk (mAP@50) Denge Grafiği ---
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)
    sns.scatterplot(
        data=df_summary, x="FPS", y="mAP@50", hue="Model",
        size="Params (M)", sizes=(160, 520), palette="deep", ax=ax, legend="brief"
    )
    for _, row in df_summary.iterrows():
        ax.annotate(
            f"{row['Model']}\n({row['mAP@50']:.3f} mAP | {row['FPS']} FPS)",
            (row["FPS"], row["mAP@50"]),
            textcoords="offset points", xytext=(0, 12), ha="center", fontsize=8.5, fontweight="bold"
        )
    ax.set_title("Doğruluk (mAP@50) vs. Çıkarım Hızı (FPS) ve Model Boyutu Analizi", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Çıkarım Hızı (FPS - Frame Per Second)")
    ax.set_ylabel("mAP@50")
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "2_hiz_dogruluk_dengesi.png", dpi=300)
    plt.close()

    # --- GRAFİK 3: Araç Sınıflarına Göre mAP@50 Karşılaştırması ---
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    df_cls_melted = df_classes.melt(id_vars="Model", value_vars=CLASS_NAMES, var_name="Sınıf", value_name="mAP@50")
    cls_plot = sns.barplot(data=df_cls_melted, x="Sınıf", y="mAP@50", hue="Model", palette="Paired", ax=ax)
    ax.set_title("Araç Sınıflarına Göre Model Başarım Karşılaştırması (mAP@50)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mAP@50")
    ax.set_xlabel("Hedef Sınıflar")
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")

    for container in cls_plot.containers:
        cls_plot.bar_label(container, fmt="%.2f", fontsize=7, padding=3, rotation=90)

    plt.tight_layout()
    plt.savefig(output_dir / "3_sinif_bazli_karsilastirma.png", dpi=300)
    plt.close()

    # --- GRAFİK 4: Tüm Modellerin Normalize Edilmiş Karmaşıklık Matrisi (Confusion Matrix) ---
    valid_cms = {k: v for k, v in conf_matrices.items() if v is not None}
    if valid_cms:
        n_models = len(valid_cms)
        cols = 3
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.2 * rows), dpi=300)
        axes = np.array(axes).reshape(-1)
        cm_labels = CLASS_NAMES + ["background"]

        for idx, (m_name, cm) in enumerate(valid_cms.items()):
            ax = axes[idx]
            col_sums = cm.sum(axis=0, keepdims=True)
            cm_norm = np.divide(cm, col_sums, out=np.zeros_like(cm, dtype=float), where=col_sums != 0)
            sns.heatmap(
                cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=cm_labels, yticklabels=cm_labels, cbar=False, ax=ax,
                annot_kws={"size": 8}
            )
            ax.set_title(m_name, fontsize=11, fontweight="bold")
            ax.set_xlabel("Gerçek Sınıf (Ground Truth)", fontsize=9)
            ax.set_ylabel("Tahmin Edilen Sınıf", fontsize=9)
            ax.tick_params(axis='both', labelsize=8)

        for j in range(idx + 1, len(axes)):
            axes[j].axis("off")

        plt.suptitle("Modellerin Normalize Edilmiş Karmaşıklık Matrisleri (Confusion Matrix)", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / "4_karmasiklik_matrisleri.png", dpi=300, bbox_inches="tight")
        plt.close()


# ==============================================================================
# 6. ANA ÇALIŞTIRMA BLOĞU
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
    all_class_ap50 = []
    all_detailed_classes = []
    conf_matrices = {}

    for model_name, weight_path in MODELS.items():
        summary, class_ap50_row, detailed_rows, cm = evaluate_model(model_name, weight_path, yaml_path)
        if summary is not None:
            all_summaries.append(summary)
            all_class_ap50.append(class_ap50_row)
            all_detailed_classes.extend(detailed_rows)
            conf_matrices[model_name] = cm

    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        df_classes = pd.DataFrame(all_class_ap50)[["Model"] + CLASS_NAMES]
        df_detailed = pd.DataFrame(all_detailed_classes)

        print("\n" + "="*90)
        print("NİHAİ MODEL VE ABLATION KARŞILAŞTIRMA TABLOSU")
        print("="*90)
        print(df_summary.to_string(index=False))

        print("\n" + "="*90)
        print("SINIF BAZLI mAP@50 KARŞILAŞTIRMA TABLOSU")
        print("="*90)
        print(df_classes.to_string(index=False))

        df_summary.to_csv(OUTPUT_DIR / "genel_sonuclar.csv", index=False)
        df_classes.to_csv(OUTPUT_DIR / "sinif_bazli_map50.csv", index=False)
        df_detailed.to_csv(OUTPUT_DIR / "sinif_bazli_detayli_metrikler.csv", index=False)

        plot_all_comparisons(df_summary, df_classes, conf_matrices, OUTPUT_DIR)
        print(f"\nTüm tablolar, PR/AUC eğrileri, Karmaşıklık Matrisleri ve 300 DPI grafikler kaydedildi:\n-> {OUTPUT_DIR}")