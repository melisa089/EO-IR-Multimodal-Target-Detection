import copy
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.dataset import YOLODataset

# Yerel veri seti ve ağırlık yolları
DATASET_DIR = Path(r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Preprocessed_Fusion")
BASE_DIR = Path(r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\YOLOv8s_Fusion")
BEST_WEIGHTS = BASE_DIR / "results" / "mid_fusion_yolov8s" / "weights" / "best.pt"
YAML_PATH = BASE_DIR / "fusion_yolov8_test.yaml"


# --- MİMARİ SINIFLARI ---
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


class TwoStreamYOLOv8MidFusion(nn.Module):
    def __init__(self, base_yolo_model):
        super().__init__()
        self.rgb_backbone = copy.deepcopy(base_yolo_model.model[:10])
        self.ir_backbone = copy.deepcopy(base_yolo_model.model[:10])
        self.fuse_p3 = MidFusionBlock(128)
        self.fuse_p4 = MidFusionBlock(256)
        self.fuse_p5 = MidFusionBlock(512)
        self.add_module("model", copy.deepcopy(base_yolo_model.model[10:]))

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        rgb_feats = {}
        h_rgb = x_rgb
        for idx, layer in enumerate(self.rgb_backbone):
            h_rgb = layer(h_rgb)
            if idx in (4, 6, 9):
                rgb_feats[idx] = h_rgb

        ir_feats = {}
        h_ir = x_ir
        for idx, layer in enumerate(self.ir_backbone):
            h_ir = layer(h_ir)
            if idx in (4, 6, 9):
                ir_feats[idx] = h_ir

        p3 = self.fuse_p3(rgb_feats[4], ir_feats[4])
        p4 = self.fuse_p4(rgb_feats[6], ir_feats[6])
        p5 = self.fuse_p5(rgb_feats[9], ir_feats[9])

        y = {4: p3, 6: p4, 9: p5}
        h = p5
        for idx, layer in enumerate(self.model, start=10):
            if layer.f != -1:
                h = y[layer.f] if isinstance(layer.f, int) else [h if j == -1 else y[j] for j in layer.f]
            h = layer(h)
            y[idx] = h

        return h

    def fuse(self, verbose=True):
        return self


# --- MODALİTE İZOLE VERİ SETİ VE DOĞRULAYICI ---
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


def run_eval(best_model, mode: str, save_folder: str):
    val_args = dict(
        model=str(BEST_WEIGHTS),
        data=str(YAML_PATH),
        split="test",
        imgsz=640,
        batch=8,
        workers=0,  # Alt işlemlerde modalite sıfırlanmasını kesin olarak önler
        device=0,
        iou=0.55,
        project=str(BASE_DIR / "results"),
        name=save_folder,
        exist_ok=True
    )
    validator = MidFusionValidator(args=val_args, modality=mode)
    metrics_dict = validator(model=best_model)
    return metrics_dict


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Veri seti bulunamadı: {DATASET_DIR}")

    rgb_folder = "images" if (DATASET_DIR / "test" / "images").exists() else "images_rgb"
    cfg = {
        "path": str(DATASET_DIR),
        "train": f"train/{rgb_folder}",
        "val": f"val/{rgb_folder}",
        "test": f"test/{rgb_folder}",
        "channels": 6,
        "nc": 5,
        "names": ["car", "truck", "bus", "van", "freight car"]
    }
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    best_ckpt = torch.load(str(BEST_WEIGHTS), map_location="cuda:0", weights_only=False)
    best_model = best_ckpt["model"].half().eval()

    print(f"Model Sınıfı: {type(best_model).__name__}")
    print(f"Toplam Parametre: {sum(p.numel() for p in best_model.parameters()):,}")

    print("\n=== 1. TAM FÜZYON TEST SETİ SONUÇLARI (RGB + IR) ===")
    m_both = run_eval(best_model, "both", "test_eval_fusion_both")

    print("\n=== 2. ABLASYON TESTİ: SADECE RGB AKTİF (IR = 0) ===")
    m_rgb = run_eval(best_model, "rgb_only", "test_eval_ablation_rgb_only")

    print("\n=== 3. ABLASYON TESTİ: SADECE IR AKTİF (RGB = 0) ===")
    m_ir = run_eval(best_model, "ir_only", "test_eval_ablation_ir_only")

    print("\n================ ÖZET KANIT TABLOSU ================")
    for label, res in [
        ("RGB + IR (Tam Füzyon)", m_both),
        ("Sadece RGB (IR Kapalı)", m_rgb),
        ("Sadece IR (RGB Kapalı)", m_ir),
    ]:
        prec = res.get("metrics/precision(B)", 0.0)
        rec = res.get("metrics/recall(B)", 0.0)
        map50 = res.get("metrics/mAP50(B)", 0.0)
        map50_95 = res.get("metrics/mAP50-95(B)", 0.0)
        print(f"{label:<24} -> P: {prec:.4f} | R: {rec:.4f} | mAP@0.5: {map50:.4f} | mAP@0.5:0.95: {map50_95:.4f}")


if __name__ == "__main__":
    main()