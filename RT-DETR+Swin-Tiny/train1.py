import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torchvision.ops import box_iou
from transformers import RTDetrForObjectDetection, RTDetrConfig
import timm

# ==============================================================================
# 1. DİĞER MODELLERLE BİREBİR AYNI DENEY ŞARTLARI
# ==============================================================================
SEED = 42
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 1          # 4 GB VRAM için fiziksel batch = 1
GRAD_ACCUM = 16         # 1 * 16 = 16 (Diğer eğitimlerindeki nbs=16 ile birebir aynı efektif batch)
LR0 = 1e-4              # ViT / Transformer mimarileri için AdamW başlangıç öğrenme oranı
WEIGHT_DECAY = 1e-4
IOU_THRESH = 0.55       # Diğer eğitimlerindeki iou=0.55 şartı
NUM_CLASSES = 5
CLASS_NAMES = ["car", "truck", "bus", "van", "freight car"]

CANDIDATE_DIRS = [
    Path(r"C:\Users\melis\Desktop\BitirmeModelEgitimleri\Dataset_Preprocessed_Fusion"),
    Path(r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Preprocessed_Fusion"),
]
DATASET_DIR = next((p for p in CANDIDATE_DIRS if p.exists()), CANDIDATE_DIRS[0])

WORK_DIR = Path(__file__).resolve().parent
SAVE_PROJECT_DIR = WORK_DIR / "results"
RUN_NAME = "mid_fusion_swin_tiny_vit_cross_attention"
RUN_DIR = SAVE_PROJECT_DIR / RUN_NAME
WEIGHTS_DIR = RUN_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 2. SENKRONİZE AUGMENTATION DESTEKLİ MULTIMODAL DATASET
# ==============================================================================
class MultimodalViTDataset(Dataset):
    def __init__(self, dataset_dir: Path, split: str = "train", img_size: int = 640, augment: bool = False, modality: str = "both"):
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.modality = modality

        rgb_folder = "images" if (dataset_dir / split / "images").exists() else "images_rgb"
        self.rgb_dir = dataset_dir / split / rgb_folder
        self.ir_dir = dataset_dir / split / "images_ir"
        self.lbl_dir = dataset_dir / split / "labels"

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.rgb_files = sorted([p for p in self.rgb_dir.iterdir() if p.suffix.lower() in exts])
        if len(self.rgb_files) == 0:
            raise FileNotFoundError(f"{self.rgb_dir} içinde görüntü bulunamadı!")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.rgb_files)

    def _apply_sync_augment(self, im_rgb, im_ir, boxes):
        if random.random() < 0.5:
            im_rgb = np.fliplr(im_rgb).copy()
            im_ir = np.fliplr(im_ir).copy()
            if len(boxes) > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]

        if random.random() < 0.5:
            im_rgb = np.flipud(im_rgb).copy()
            im_ir = np.flipud(im_ir).copy()
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]

        if random.random() < 0.5:
            angle = random.uniform(-15.0, 15.0)
            scale = random.uniform(0.75, 1.25)
            h, w = self.img_size, self.img_size
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            im_rgb = cv2.warpAffine(im_rgb, M, (w, h), borderValue=(114, 114, 114))
            im_ir = cv2.warpAffine(im_ir, M, (w, h), borderValue=(114, 114, 114))

            if len(boxes) > 0:
                new_boxes = []
                valid_mask = []
                for box in boxes:
                    cx, cy, bw, bh = box
                    x1, y1 = (cx - bw / 2.0) * w, (cy - bh / 2.0) * h
                    x2, y2 = (cx + bw / 2.0) * w, (cy + bh / 2.0) * h
                    corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]])
                    trans = corners @ M.T
                    nx1, ny1 = np.clip(trans[:, 0].min(), 0, w), np.clip(trans[:, 1].min(), 0, h)
                    nx2, ny2 = np.clip(trans[:, 0].max(), 0, w), np.clip(trans[:, 1].max(), 0, h)
                    nw, nh = (nx2 - nx1) / w, (ny2 - ny1) / h
                    if nw > 0.005 and nh > 0.005:
                        new_boxes.append([(nx1 + nx2) / (2.0 * w), (ny1 + ny2) / (2.0 * h), nw, nh])
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                boxes = np.array(new_boxes, dtype=np.float32) if new_boxes else np.zeros((0, 4), dtype=np.float32)
                return im_rgb, im_ir, boxes, valid_mask

        return im_rgb, im_ir, boxes, [True] * len(boxes)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        ir_path = self.ir_dir / rgb_path.name
        if not ir_path.exists():
            matches = list(self.ir_dir.glob(f"{rgb_path.stem}.*"))
            ir_path = matches[0] if matches else rgb_path

        im_rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        ir_raw = cv2.imread(str(ir_path))
        im_ir = cv2.cvtColor(ir_raw, cv2.COLOR_BGR2RGB) if ir_raw is not None else im_rgb.copy()

        im_rgb = cv2.resize(im_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        im_ir = cv2.resize(im_ir, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        lbl_path = self.lbl_dir / f"{rgb_path.stem}.txt"
        cls_list, box_list = [], []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        c = int(float(parts[0]))
                        cx, cy, bw, bh = map(float, parts[1:5])
                        if bw > 0 and bh > 0:
                            cls_list.append(c)
                            box_list.append([cx, cy, bw, bh])

        boxes = np.array(box_list, dtype=np.float32) if box_list else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(cls_list, dtype=np.int64) if cls_list else np.zeros((0,), dtype=np.int64)

        if self.augment:
            im_rgb, im_ir, boxes, valid_mask = self._apply_sync_augment(im_rgb, im_ir, boxes)
            if len(labels) > 0:
                labels = labels[valid_mask]

        if self.modality == "rgb_only":
            im_ir = np.zeros_like(im_ir)
        elif self.modality == "ir_only":
            im_rgb = np.zeros_like(im_rgb)

        t_rgb = torch.from_numpy(((im_rgb.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1))
        t_ir = torch.from_numpy(((im_ir.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1))

        return t_rgb, t_ir, {"class_labels": torch.from_numpy(labels), "boxes": torch.from_numpy(boxes)}


def collate_fn(batch):
    return (
        torch.stack([b[0] for b in batch]),
        torch.stack([b[1] for b in batch]),
        [b[2] for b in batch],
    )


# ==============================================================================
# 3. ÇİFT YÖNLÜ CROSS-ATTENTION MID-LEVEL FUSION MODÜLÜ (VRAM OPTİMİZE)
# ==============================================================================
class CrossAttentionMidFusionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, sr_ratio=1):
        super().__init__()
        self.sr_ratio = sr_ratio
        self.attn_rgb_ir = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.attn_ir_rgb = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm_rgb = nn.LayerNorm(channels)
        self.norm_ir = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)
        )

    def _forward_impl(self, f_rgb, f_ir):
        b, c, h, w = f_rgb.shape
        if self.sr_ratio > 1:
            r_rgb = F.avg_pool2d(f_rgb, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            r_ir = F.avg_pool2d(f_ir, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            rh, rw = r_rgb.shape[2], r_rgb.shape[3]
        else:
            r_rgb, r_ir = f_rgb, f_ir
            rh, rw = h, w

        q_rgb = r_rgb.flatten(2).transpose(1, 2)
        q_ir = r_ir.flatten(2).transpose(1, 2)

        out_rgb, _ = self.attn_rgb_ir(query=q_rgb, key=q_ir, value=q_ir)
        out_rgb = self.norm_rgb(q_rgb + out_rgb).transpose(1, 2).view(b, c, rh, rw)

        out_ir, _ = self.attn_ir_rgb(query=q_ir, key=q_rgb, value=q_rgb)
        out_ir = self.norm_ir(q_ir + out_ir).transpose(1, 2).view(b, c, rh, rw)

        if self.sr_ratio > 1:
            out_rgb = f_rgb + F.interpolate(out_rgb, size=(h, w), mode="bilinear", align_corners=False)
            out_ir = f_ir + F.interpolate(out_ir, size=(h, w), mode="bilinear", align_corners=False)

        return self.proj(torch.cat([out_rgb, out_ir], dim=1))

    def forward(self, f_rgb, f_ir):
        if self.training and (f_rgb.requires_grad or f_ir.requires_grad):
            return checkpoint(self._forward_impl, f_rgb, f_ir, use_reentrant=False)
        return self._forward_impl(f_rgb, f_ir)


# ==============================================================================
# 4. ÇİFT OMURGALI SWIN-TINY ViT + CROSS-ATTENTION BACKBONE MODÜLÜ
# ==============================================================================
class DualSwinCrossAttnBackbone(nn.Module):
    def __init__(self, img_size=640):
        super().__init__()
        self.rgb_backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=img_size
        )
        self.ir_backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3),
            img_size=img_size
        )
        # 4 GB VRAM tasarrufu için Gradient Checkpointing aktif
        self.rgb_backbone.set_grad_checkpointing(True)
        self.ir_backbone.set_grad_checkpointing(True)

        # P3 (80x80 -> 20x20 token), P4 (40x40 -> 20x20 token), P5 (20x20 -> 20x20 token)
        self.fuse_p3 = CrossAttentionMidFusionBlock(channels=192, num_heads=4, sr_ratio=4)
        self.fuse_p4 = CrossAttentionMidFusionBlock(channels=384, num_heads=4, sr_ratio=2)
        self.fuse_p5 = CrossAttentionMidFusionBlock(channels=768, num_heads=8, sr_ratio=1)
        self.current_ir = None

    def forward(self, pixel_values, pixel_mask=None):
        feats_rgb = self.rgb_backbone(pixel_values)
        feats_ir = self.ir_backbone(self.current_ir)

        f_rgb = [f.permute(0, 3, 1, 2).contiguous() for f in feats_rgb]
        f_ir = [f.permute(0, 3, 1, 2).contiguous() for f in feats_ir]

        p3 = self.fuse_p3(f_rgb[0], f_ir[0])
        p4 = self.fuse_p4(f_rgb[1], f_ir[1])
        p5 = self.fuse_p5(f_rgb[2], f_ir[2])

        out = []
        for feat in (p3, p4, p5):
            mask = torch.zeros((feat.shape[0], feat.shape[2], feat.shape[3]), dtype=torch.bool, device=feat.device)
            out.append((feat, mask))
        return out


class TwoStreamSwinViTMidFusion(nn.Module):
    def __init__(self, num_classes=5, img_size=640):
        super().__init__()
        config = RTDetrConfig(
            num_labels=num_classes,
            encoder_in_channels=[192, 384, 768],
            encoder_hidden_dim=256,
            d_model=256,
            num_queries=150,
            use_pretrained_backbone=False,
        )
        self.detector = RTDetrForObjectDetection(config)
        self.detector.model.backbone = DualSwinCrossAttnBackbone(img_size=img_size)

        hidden_dim = config.encoder_hidden_dim
        self.detector.model.encoder_input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
            )
            for in_ch in [192, 384, 768]
        ])

    def forward(self, x_rgb, x_ir, labels=None):
        self.detector.model.backbone.current_ir = x_ir
        return self.detector(pixel_values=x_rgb, labels=labels)


# ==============================================================================
# 5. SAF PYTORCH COCO mAP & METRİK HESAPLAYICI (EK PAKET GEREKTİRMEZ)
# ==============================================================================
def compute_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@torch.no_grad()
def evaluate_split(model, loader, device, iou_thresh=0.55, conf_thresh=0.25):
    model.eval()
    iou_thresholds = np.linspace(0.50, 0.95, 10)
    all_preds = {c: [] for c in range(NUM_CLASSES)}
    all_gts = {c: {} for c in range(NUM_CLASSES)}
    gt_counts = {c: 0 for c in range(NUM_CLASSES)}

    tp_pr, fp_pr, fn_pr = 0, 0, 0
    latencies = []
    img_counter = 0

    for rgb, ir, targets in loader:
        rgb, ir = rgb.to(device), ir.to(device)
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(rgb, ir)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0 / rgb.shape[0])

        logits = outputs.logits.softmax(-1)[..., :-1]
        scores, pred_labels = logits.max(-1)
        pred_boxes = outputs.pred_boxes

        for i in range(rgb.shape[0]):
            keep = scores[i] > 0.01
            b = pred_boxes[i][keep].cpu()
            s = scores[i][keep].cpu()
            l = pred_labels[i][keep].cpu()

            xyxy = torch.zeros_like(b)
            if len(b) > 0:
                xyxy[:, 0] = (b[:, 0] - b[:, 2] / 2.0) * IMG_SIZE
                xyxy[:, 1] = (b[:, 1] - b[:, 3] / 2.0) * IMG_SIZE
                xyxy[:, 2] = (b[:, 0] + b[:, 2] / 2.0) * IMG_SIZE
                xyxy[:, 3] = (b[:, 1] + b[:, 3] / 2.0) * IMG_SIZE

            gt_b = targets[i]["boxes"].cpu()
            gt_l = targets[i]["class_labels"].cpu()
            gt_xyxy = torch.zeros_like(gt_b)
            if len(gt_b) > 0:
                gt_xyxy[:, 0] = (gt_b[:, 0] - gt_b[:, 2] / 2.0) * IMG_SIZE
                gt_xyxy[:, 1] = (gt_b[:, 1] - gt_b[:, 3] / 2.0) * IMG_SIZE
                gt_xyxy[:, 2] = (gt_b[:, 0] + gt_b[:, 2] / 2.0) * IMG_SIZE
                gt_xyxy[:, 3] = (gt_b[:, 1] + gt_b[:, 3] / 2.0) * IMG_SIZE

            for c in range(NUM_CLASSES):
                c_gt = gt_xyxy[gt_l == c]
                if len(c_gt) > 0:
                    all_gts[c][img_counter] = c_gt
                    gt_counts[c] += len(c_gt)

                c_mask = l == c
                for box_p, conf_p in zip(xyxy[c_mask], s[c_mask]):
                    all_preds[c].append((float(conf_p.item()), img_counter, box_p))

            valid_p = s > conf_thresh
            p_boxes_c, p_cls_c = xyxy[valid_p], l[valid_p]
            if len(gt_xyxy) == 0:
                fp_pr += len(p_boxes_c)
            elif len(p_boxes_c) == 0:
                fn_pr += len(gt_xyxy)
            else:
                ious = box_iou(p_boxes_c, gt_xyxy)
                matched_gt = set()
                for idx_p in range(len(p_boxes_c)):
                    best_iou, best_j = 0.0, -1
                    for idx_g in range(len(gt_xyxy)):
                        if idx_g in matched_gt or p_cls_c[idx_p] != gt_l[idx_g]:
                            continue
                        val_iou = float(ious[idx_p, idx_g].item())
                        if val_iou > best_iou:
                            best_iou, best_j = val_iou, idx_g
                    if best_iou >= iou_thresh:
                        tp_pr += 1
                        matched_gt.add(best_j)
                    else:
                        fp_pr += 1
                fn_pr += len(gt_xyxy) - len(matched_gt)

            img_counter += 1

    aps_per_iou = {t: [] for t in iou_thresholds}
    for c in range(NUM_CLASSES):
        if gt_counts[c] == 0:
            continue
        preds_c = sorted(all_preds[c], key=lambda x: x[0], reverse=True)
        if len(preds_c) == 0:
            for t in iou_thresholds:
                aps_per_iou[t].append(0.0)
            continue

        for t in iou_thresholds:
            matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in all_gts[c].items()}
            tp = np.zeros(len(preds_c))
            fp = np.zeros(len(preds_c))

            for idx_p, (_, img_id, p_box) in enumerate(preds_c):
                if img_id not in all_gts[c]:
                    fp[idx_p] = 1.0
                    continue
                g_boxes = all_gts[c][img_id]
                ious = box_iou(p_box.unsqueeze(0), g_boxes).squeeze(0).numpy()
                best_g = int(np.argmax(ious))
                if ious[best_g] >= t and not matched[img_id][best_g]:
                    tp[idx_p] = 1.0
                    matched[img_id][best_g] = True
                else:
                    fp[idx_p] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / (gt_counts[c] + 1e-16)
            prec = cum_tp / (cum_tp + cum_fp + 1e-16)
            aps_per_iou[t].append(compute_ap(rec, prec))

    map50 = float(np.mean(aps_per_iou[iou_thresholds[0]])) if aps_per_iou[iou_thresholds[0]] else 0.0
    map50_95 = float(np.mean([np.mean(aps_per_iou[t]) for t in iou_thresholds if aps_per_iou[t]]))
    avg_ms = float(np.mean(latencies)) if latencies else 0.0

    return {
        "P": tp_pr / (tp_pr + fp_pr + 1e-16),
        "R": tp_pr / (tp_pr + fn_pr + 1e-16),
        "mAP50": map50,
        "mAP50_95": map50_95,
        "fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0
    }


# ==============================================================================
# 6. ANA EĞİTİM VE TEST ABLASYON AKIŞI
# ==============================================================================
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU bulunamadı!")
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Veri seti klasörü bulunamadı: {DATASET_DIR}")

    seed_everything(SEED)
    torch.cuda.empty_cache()

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Kullanılan GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"Model: Two-Stream Swin-Tiny ViT + Cross-Attention + RT-DETR")
    print(f"Veri Seti Yolu: {DATASET_DIR}")
    print(f"Sonuç Klasörü: {RUN_DIR}")

    train_ds = MultimodalViTDataset(DATASET_DIR, split="train", img_size=IMG_SIZE, augment=True, modality="both")
    val_ds = MultimodalViTDataset(DATASET_DIR, split="val", img_size=IMG_SIZE, augment=False, modality="both")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)

    model = TwoStreamSwinViTMidFusion(num_classes=NUM_CLASSES, img_size=IMG_SIZE).to(device)
    print(f"Toplam Parametre: {sum(p.numel() for p in model.parameters()):,}")

    # İlk 15 epoch VRAM ve kararlılık için erken katmanları dondur
    for name, p in model.named_parameters():
        if ("rgb_backbone" in name or "ir_backbone" in name) and ("layers.0" in name or "layers.1" in name):
            p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR0, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR0 * 0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=True)

    best_map50_95 = -1.0
    best_weights = WEIGHTS_DIR / "best.pt"
    log_csv = RUN_DIR / "results.csv"

    with open(log_csv, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_P,val_R,val_mAP50,val_mAP50_95\n")

    for epoch in range(1, EPOCHS + 1):
        if epoch == 16:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR0 * 0.5, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 15, eta_min=LR0 * 0.01)
            print("--> [Bilgi] Swin-Tiny omurga katmanlarının tamamı eğitime açıldı.")

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{EPOCHS}")
        for step, (rgb, ir, targets) in pbar:
            rgb, ir = rgb.to(device), ir.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(rgb, ir, labels=targets)
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += outputs.loss.item()
            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        if epoch % 5 == 0 or epoch >= (EPOCHS - 15) or epoch == 1:
            torch.cuda.empty_cache()
            val_metrics = evaluate_split(model, val_loader, device, iou_thresh=IOU_THRESH)
            print(
                f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
                f"P: {val_metrics['P']:.4f} | R: {val_metrics['R']:.4f} | "
                f"mAP@0.5: {val_metrics['mAP50']:.4f} | mAP@0.5:0.95: {val_metrics['mAP50_95']:.4f}"
            )
            with open(log_csv, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{avg_loss:.4f},{val_metrics['P']:.4f},{val_metrics['R']:.4f},{val_metrics['mAP50']:.4f},{val_metrics['mAP50_95']:.4f}\n")

            if val_metrics["mAP50_95"] > best_map50_95:
                best_map50_95 = val_metrics["mAP50_95"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "best_map50_95": best_map50_95}, best_weights)
                print(f"    [EN İYİ AĞIRLIK KAYDEDİLDİ] mAP@0.5:0.95 = {best_map50_95:.4f}")
            torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- SWIN-TINY ViT (CROSS-ATTENTION) TEST SETİ VE ABLASYON SONUÇLARI ---")
    ckpt = torch.load(str(best_weights), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    print("\n================ SWIN-TINY ViT ÖZET KANIT TABLOSU ================")
    for label, mod_mode in [
        ("RGB + IR (Tam Füzyon)", "both"),
        ("Sadece RGB (IR Kapalı)", "rgb_only"),
        ("Sadece IR (RGB Kapalı)", "ir_only"),
    ]:
        test_ds = MultimodalViTDataset(DATASET_DIR, split="test", img_size=IMG_SIZE, augment=False, modality=mod_mode)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        res = evaluate_split(model, test_loader, device, iou_thresh=IOU_THRESH)
        print(
            f"{label:<24} -> P: {res['P']:.4f} | R: {res['R']:.4f} | "
            f"mAP@0.5: {res['mAP50']:.4f} | mAP@0.5:0.95: {res['mAP50_95']:.4f} | "
            f"FPS: {res['fps']:.1f}"
        )


if __name__ == "__main__":
    main()