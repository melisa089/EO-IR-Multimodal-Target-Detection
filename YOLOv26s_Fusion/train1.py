import gc
import copy
import yaml
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER

# Veri seti yolu ve çalışma ortamı ayarları (YOLOv8s ve YOLO11s ile birebir aynı koşullar)
DATASET_DIR = Path(r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Preprocessed_Fusion")
WORK_DIR = Path(__file__).resolve().parent
YAML_PATH = WORK_DIR / "fusion_yolo26s.yaml"
SAVE_PROJECT_DIR = WORK_DIR / "results"
RUN_NAME = "mid_fusion_yolo26s"
MODEL_WEIGHTS = "yolo26s.pt"


# 1. Kanal Dikkat (Channel Attention) Destekli Mid-Level Fusion Modülü
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


# 2. Dinamik Katman Tespiti Yapan Çift Omurgalı YOLO26s Mid-Level Fusion Mimarisi
class TwoStreamYOLO26MidFusion(DetectionModel):
    def __init__(self, base_yolo_model):
        nn.Module.__init__(self)
        self.yaml = copy.deepcopy(base_yolo_model.yaml)
        self.yaml["ch"] = 6
        self.names = copy.deepcopy(base_yolo_model.names)
        self.nc = base_yolo_model.nc
        self.stride = base_yolo_model.stride.clone()
        self.args = copy.deepcopy(base_yolo_model.args)
        self.inplace = True

        # Omurga bitişini ve P3, P4, P5 füzyon indekslerini model yapısından otomatik tespit et
        upsample_idx = next(
            i for i, m in enumerate(base_yolo_model.model)
            if "Upsample" in m.__class__.__name__
        )
        backbone_end = upsample_idx

        skip_indices = []
        for i in range(backbone_end, len(base_yolo_model.model) - 1):
            f = base_yolo_model.model[i].f
            if isinstance(f, list):
                for idx in f:
                    if 0 <= idx < backbone_end and idx not in skip_indices:
                        skip_indices.append(idx)

        p5_idx = backbone_end - 1
        skip_indices = sorted([idx for idx in skip_indices if idx != p5_idx])
        self.fuse_indices = tuple(skip_indices + [p5_idx])  # Tam olarak (4, 6, 10)
        self.backbone_end = backbone_end

        LOGGER.info(f"Tespit edilen omurga katman sayısı: {backbone_end} | Füzyon indeksleri (P3, P4, P5): {self.fuse_indices}")

        # ÖNCE self.model atanmalı, SONRA self.end2end ayarlanmalı (Ultralytics 8.4+ setter uyumu)
        self.model = copy.deepcopy(base_yolo_model.model[backbone_end:])
        self.end2end = getattr(base_yolo_model, "end2end", False)

        # İki ayrı pretrained omurga (RGB ve IR)
        self.rgb_backbone = copy.deepcopy(base_yolo_model.model[:backbone_end])
        self.ir_backbone = copy.deepcopy(base_yolo_model.model[:backbone_end])

        # Dummy ileri geçiş (forward) ile P3, P4, P5 kanal sayılarını hatasız ölç
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 64, 64)
            feats = {}
            h = dummy
            for idx, layer in enumerate(self.rgb_backbone):
                h = layer(h)
                if idx in self.fuse_indices:
                    feats[idx] = h.shape[1]

        c_p3, c_p4, c_p5 = [feats[idx] for idx in self.fuse_indices]
        LOGGER.info(f"Füzyon kanal boyutları -> P3: {c_p3}, P4: {c_p4}, P5: {c_p5}")

        self.fuse_p3 = MidFusionBlock(c_p3)
        self.fuse_p4 = MidFusionBlock(c_p4)
        self.fuse_p5 = MidFusionBlock(c_p5)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        x_rgb = x[:, :3, :, :]
        x_ir = x[:, 3:, :, :]

        idx_p3, idx_p4, idx_p5 = self.fuse_indices

        rgb_feats = {}
        h_rgb = x_rgb
        for idx, layer in enumerate(self.rgb_backbone):
            h_rgb = layer(h_rgb)
            if idx in self.fuse_indices:
                rgb_feats[idx] = h_rgb

        ir_feats = {}
        h_ir = x_ir
        for idx, layer in enumerate(self.ir_backbone):
            h_ir = layer(h_ir)
            if idx in self.fuse_indices:
                ir_feats[idx] = h_ir

        p3 = self.fuse_p3(rgb_feats[idx_p3], ir_feats[idx_p3])
        p4 = self.fuse_p4(rgb_feats[idx_p4], ir_feats[idx_p4])
        p5 = self.fuse_p5(rgb_feats[idx_p5], ir_feats[idx_p5])

        y = {idx_p3: p3, idx_p4: p4, idx_p5: p5}
        h = p5
        for idx, layer in enumerate(self.model, start=self.backbone_end):
            if layer.f != -1:
                h = y[layer.f] if isinstance(layer.f, int) else [h if j == -1 else y[j] for j in layer.f]
            h = layer(h)
            y[idx] = h

        return h


# 3. RGB ve IR Çiftini Senkronize Yükleyen 6-Kanallı Veri Seti
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


# 4. Özel Multimodal Validator ve Trainer
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


class MidFusionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        base = YOLO(MODEL_WEIGHTS).model
        fusion_model = TwoStreamYOLO26MidFusion(base)
        del base
        gc.collect()
        torch.cuda.empty_cache()
        LOGGER.info("4GB VRAM Optimize Two-Stream YOLO26s Mid-Level Fusion mimarisi yüklendi.")
        return fusion_model

    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(self.model.stride.max() if self.model else 32), 32)
        return MultimodalYOLODataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect if mode == "val" else False,
            cache=False,
            single_cls=self.args.single_cls,
            stride=gs,
            pad=0.0 if mode == "train" else 0.5,
            prefix="",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            modality="both"
        )

    def get_validator(self):
        return MidFusionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy.deepcopy(self.args), _callbacks=self.callbacks, modality="both"
        )


def run_test_eval(best_weights: Path, best_model, base_args: dict, mode: str, folder_name: str):
    val_args = copy.deepcopy(base_args)
    val_args["model"] = str(best_weights)
    val_args["split"] = "test"
    val_args["batch"] = 8
    val_args["workers"] = 0
    val_args["project"] = str(SAVE_PROJECT_DIR)
    val_args["name"] = folder_name
    val_args["exist_ok"] = True
    validator = MidFusionValidator(args=val_args, modality=mode)
    return validator(model=best_model)


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

    rgb_folder_name = "images" if (DATASET_DIR / "train" / "images").exists() else "images_rgb"
    cfg = {
        "path": str(DATASET_DIR),
        "train": f"train/{rgb_folder_name}",
        "val": f"val/{rgb_folder_name}",
        "test": f"test/{rgb_folder_name}",
        "channels": 6,
        "nc": 5,
        "names": ["car", "truck", "bus", "van", "freight car"]
    }
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # YOLOv8s ve YOLO11s eğitimleri ile birebir aynı hiperparametreler
    args = dict(
        model=MODEL_WEIGHTS,
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
        quantize=16,
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

    trainer = MidFusionTrainer(overrides=args)
    orig_get_validator = trainer.get_validator

    def safe_get_validator():
        DetectionTrainer.get_validator(trainer)
        return orig_get_validator()

    trainer.get_validator = safe_get_validator
    trainer.train()

    # Test öncesi VRAM'i tamamen temizle
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- YOLO26s MID-LEVEL FUSION TEST SETİ VE ABLASYON SONUÇLARI ---")
    best_weights = SAVE_PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    best_ckpt = torch.load(str(best_weights), map_location="cuda:0", weights_only=False)
    best_model = best_ckpt["model"].half().eval()

    print(f"Toplam Parametre: {sum(p.numel() for p in best_model.parameters()):,}")

    m_both = run_test_eval(best_weights, best_model, args, "both", f"{RUN_NAME}_test_fusion_both")
    m_rgb = run_test_eval(best_weights, best_model, args, "rgb_only", f"{RUN_NAME}_test_ablation_rgb_only")
    m_ir = run_test_eval(best_weights, best_model, args, "ir_only", f"{RUN_NAME}_test_ablation_ir_only")

    print("\n================ YOLO26s ÖZET KANIT TABLOSU ================")
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