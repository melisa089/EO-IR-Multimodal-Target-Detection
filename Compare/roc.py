import os
import gc
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import auc
from ultralytics import YOLO
import ultralytics.data.utils as data_utils
import ultralytics.data.dataset as data_dataset
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.dataset import YOLODataset

# ==============================================================================
# 0. images_ir ETİKET EŞLEŞTİRME PATCH'İ
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
# 2. AYARLAR VE MODEL YOLLARI
# ==============================================================================
DATASET_DIR = Path(r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Preprocessed_Fusion")
OUTPUT_DIR = Path(r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\Compare\test_final_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "YOLO26s-RGB":     r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv26s_Baseline\results\baseline_rgb_yolo26s\weights\best.pt",
    "YOLO26s-IR":      r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv26s_Baseline\results\baseline_ir_yolo26s\weights\best.pt",
    "YOLOv8s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv8s_Fusion\results\mid_fusion_yolov8s\weights\best.pt",
    "YOLO11s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv11s_Fusion\results\mid_fusion_yolo11s\weights\best.pt",
    "YOLO12s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv12s_Fusion\results\mid_fusion_yolo12s\weights\best.pt",
    "YOLO26s-Fusion":  r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv26s_Fusion\results\mid_fusion_yolo26s\weights\best.pt",
}

COLORS = {
    "YOLO26s-RGB":    "#66c2a5",
    "YOLO26s-IR":     "#fc8d62",
    "YOLOv8s-Fusion": "#8da0cb",
    "YOLO11s-Fusion": "#e78ac3",
    "YOLO12s-Fusion": "#a6d854",
    "YOLO26s-Fusion": "#d95f02",
}

CLASS_NAMES = ['car', 'truck', 'bus', 'van', 'freight_car']
IMG_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0 if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# 3. MULTIMODAL DATASET VE VALIDATOR
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
        elif (im_ir.shape[0], im_ir.shape[1]) != (im_rgb.shape[0], im_rgb.shape[1]):
            im_ir = cv2.resize(im_ir, (im_rgb.shape[1], im_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

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
# 4. VALIDATOR CURVES_RESULTS ÜZERİNDEN MAKRO ROC-AUC HESAPLAMA
# ==============================================================================
def get_macro_roc(model_name, weight_path, yaml_path):
    print(f"\n>>> ROC-AUC Hesaplanıyor: {model_name}")
    name_lower = model_name.lower()
    is_unimodal = ("rgb" in name_lower or "ir" in name_lower) and ("fusion" not in name_lower)

    ckpt = torch.load(weight_path, map_location="cuda:0" if torch.cuda.is_available() else "cpu", weights_only=False)
    model = ckpt["model"].half().eval() if torch.cuda.is_available() else ckpt["model"].float().eval()

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
        name=f"roc_{model_name}",
        exist_ok=True,
        plots=False,
        verbose=False
    )

    if is_unimodal:
        unimodal_yaml = OUTPUT_DIR / f"data_roc_{model_name.lower()}.yaml"
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
        box_metrics = res_obj.box
        cm_matrix = res_obj.confusion_matrix.matrix
    else:
        validator = MidFusionValidator(args=val_args, modality="both")
        validator(model=model)
        box_metrics = validator.metrics.box
        cm_matrix = validator.confusion_matrix.matrix

    # curves_results[2]: (conf_grid, precision_per_class [nc, 1000])
    # curves_results[3]: (conf_grid, recall_per_class [nc, 1000])
    curves = box_metrics.curves_results
    p_curves = curves[2][1]  # shape: (nc, 1000)
    r_curves = curves[3][1]  # shape: (nc, 1000)

    # Her sınıfın test kümesindeki toplam gerçek örnek sayısı (confusion matrix sütun toplamı)
    gt_counts = cm_matrix.sum(axis=0)[:len(CLASS_NAMES)]

    mean_fpr = np.linspace(0, 1, 500)
    tprs, aucs = [], []

    for c in range(len(CLASS_NAMES)):
        n_pos = max(1.0, float(gt_counts[c]))
        p_c = np.clip(p_curves[c], 1e-6, 1.0)
        r_c = np.clip(r_curves[c], 0.0, 1.0)

        # Güven eşiği yüksekten düşüğe giderken TP ve FP sayıları
        tp_c = r_c * n_pos
        fp_c = tp_c * (1.0 - p_c) / p_c

        # Maksimum FP sayısına göre normalize edilmiş FPR [0, 1] ve TPR = Recall [0, 1]
        max_fp = max(1.0, float(np.max(fp_c)))
        fpr_c = np.concatenate([[0.0], fp_c / max_fp, [1.0]])
        tpr_c = np.concatenate([[0.0], r_c, [1.0]])

        # Monoton artan sıralama (ROC eğrisi kuralı)
        sort_idx = np.argsort(fpr_c)
        fpr_c = fpr_c[sort_idx]
        tpr_c = np.maximum.accumulate(tpr_c[sort_idx])

        tprs.append(np.interp(mean_fpr, fpr_c, tpr_c))
        aucs.append(auc(fpr_c, tpr_c))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    macro_auc = float(np.mean(aucs))

    del model, ckpt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"    -> {model_name} Macro-AUC = {macro_auc:.4f}")
    return mean_fpr, mean_tpr, macro_auc


# ==============================================================================
# 5. ÇALIŞTIR VE TEK GRAFİKTE KAYDET
# ==============================================================================
if __name__ == "__main__":
    yaml_path = OUTPUT_DIR / "data_fusion_roc.yaml"
    cfg = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "channels": 6,
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(9.5, 7), dpi=300)

    for model_name, weight_path in MODELS.items():
        if not os.path.exists(weight_path):
            print(f"[UYARI] Bulunamadı: {weight_path}")
            continue
        fpr, tpr, macro_auc = get_macro_roc(model_name, weight_path, yaml_path)
        lw = 2.8 if model_name == "YOLO26s-Fusion" else 1.8
        ax.plot(
            fpr, tpr,
            color=COLORS[model_name], lw=lw,
            label=f"{model_name} (Macro-AUC = {macro_auc:.4f})"
        )

    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Rastgele Sınıflandırıcı (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Yanlış Pozitif Oranı (False Positive Rate - FPR)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Doğru Pozitif Oranı (True Positive Rate - TPR)", fontsize=11, fontweight="bold")
    ax.set_title("Modellerin Karşılaştırmalı Makro-Ortalama ROC Eğrileri (Bağımsız Test Kümesi)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="white", framealpha=0.92)

    plt.tight_layout()
    out_file = OUTPUT_DIR / "5_roc_karsilastirmasi.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"\nGrafik başarıyla kaydedildi:\n-> {out_file}")
    