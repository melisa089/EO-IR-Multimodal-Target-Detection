import os
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

# Veri seti yolu ve çalışma ortamı ayarları
DATASET_DIR = Path(r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Preprocessed_Fusion")
WORK_DIR = Path(__file__).resolve().parent
YAML_PATH = WORK_DIR / "fusion.yaml"
SAVE_PROJECT_DIR = WORK_DIR / "results"
RUN_NAME = "mid_fusion_yolov8s"


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


# 2. Ultralytics DetectionModel Tabanlı Çift Omurgalı Mid-Level Fusion Mimarisi
class TwoStreamYOLOv8MidFusion(DetectionModel):
    def __init__(self, base_yolo_model):
        nn.Module.__init__(self)
        self.yaml = copy.deepcopy(base_yolo_model.yaml)
        self.yaml["ch"] = 6
        self.names = copy.deepcopy(base_yolo_model.names)
        self.nc = base_yolo_model.nc
        self.stride = base_yolo_model.stride.clone()
        self.args = copy.deepcopy(base_yolo_model.args)
        self.inplace = True
        self.end2end = getattr(base_yolo_model, "end2end", False)

        # Ortak Neck & Head katmanları
        self.model = copy.deepcopy(base_yolo_model.model[10:])

        # İki ayrı pretrained omurga (RGB ve IR) + Füzyon Blokları
        self.rgb_backbone = copy.deepcopy(base_yolo_model.model[:10])
        self.ir_backbone = copy.deepcopy(base_yolo_model.model[:10])

        self.fuse_p3 = MidFusionBlock(128)
        self.fuse_p4 = MidFusionBlock(256)
        self.fuse_p5 = MidFusionBlock(512)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
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


# 3. RGB ve IR Çiftini Senkronize Yükleyen 6-Kanallı Veri Seti
class MultimodalYOLODataset(YOLODataset):
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

        im_6ch = np.concatenate([im_rgb, im_ir], axis=2)
        return im_6ch, (h0, w0), (h, w)


# 4. Özel Multimodal Validator ve Trainer
class MidFusionValidator(DetectionValidator):
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
            fraction=1.0
        )

    def warmup(self, imgsz=(1, 6, 640, 640)):
        super().warmup(imgsz=(1, 6, self.args.imgsz, self.args.imgsz))


class MidFusionTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        base = YOLO("yolov8s.pt").model
        fusion_model = TwoStreamYOLOv8MidFusion(base)
        del base
        gc.collect()
        torch.cuda.empty_cache()
        LOGGER.info("4GB VRAM Optimize Two-Stream YOLOv8s Mid-Level Fusion mimarisi yüklendi.")
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
            fraction=self.args.fraction if mode == "train" else 1.0
        )

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return MidFusionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy.deepcopy(self.args), _callbacks=self.callbacks
        )


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

    args = dict(
        model="yolov8s.pt",
        data=str(YAML_PATH),
        project=str(SAVE_PROJECT_DIR),
        name=RUN_NAME,
        device=0,
        workers=2,
        cache=False,
        epochs=100,
        imgsz=640,
        batch=4,
        nbs=16,
        amp=True,
        half=True,
        seed=42,
        optimizer="SGD",
        lr0=0.008,
        cos_lr=True,
        close_mosaic=15,
        mosaic=1.0,
        copy_paste=0.0,     # Hayalet kutu yığılmasını önlemek için kapalı
        erasing=0.0,        # Araç silinip etiket kalmasını önlemek için kapalı
        scale=0.5,
        degrees=15.0,       # Kutu taşmalarını ve aşırı genişlemeyi önlemek için sınırlı
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        iou=0.55
    )

    trainer = MidFusionTrainer(overrides=args)
    trainer.train()

    # Test öncesi VRAM'i tamamen temizle
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- YOLOv8s MID-LEVEL FUSION TEST SETİ SONUÇLARI ---")
    best_weights = SAVE_PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    best_ckpt = torch.load(str(best_weights), map_location="cuda:0")
    best_model = best_ckpt["model"].half().eval()

    val_args = copy.deepcopy(args)
    val_args["split"] = "test"
    val_args["batch"] = 4
    val_args["save_dir"] = str(SAVE_PROJECT_DIR / f"{RUN_NAME}_test_eval")
    validator = MidFusionValidator(args=val_args)
    validator(model=best_model)


if __name__ == "__main__":
    main()