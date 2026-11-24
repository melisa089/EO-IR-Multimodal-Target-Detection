import os
import cv2
import numpy as np
from pathlib import Path

SRC_DIR = r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Final_Balanced"
OUT_DIR = r"C:\Users\melis\Desktop\BitirmeProjesi\Dataset_Preprocessed_Fusion"

ORIG_W, ORIG_H = 840, 712
X_MIN, Y_MIN = 100, 100
CROP_W, CROP_H = 640, 512

clahe_rgb = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
clahe_ir = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def enhance_rgb(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe_rgb.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
    return cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0)

def enhance_ir(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq = clahe_ir.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def resolve_modal_folders(split_dir):
    """images_rgb ve images_ir klasörlerinin hangisinin renkli RGB, hangisinin gri/termal IR olduğunu otomatik doğrular."""
    f1 = os.path.join(split_dir, "images_rgb")
    f2 = os.path.join(split_dir, "images_ir")

    sample_files = [f for f in os.listdir(f1) if f.endswith((".jpg", ".png"))][:5]
    f1_is_colored = False
    for sf in sample_files:
        img = cv2.imread(os.path.join(f1, sf))
        if img is not None:
            if not (np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2])):
                f1_is_colored = True
                break

    if f1_is_colored:
        return f1, f2
    else:
        print(f"[BİLGİ] {os.path.basename(split_dir)}: Klasör isimleri ters bulundu, otomatik düzeltiliyor (images_ir -> RGB, images_rgb -> IR).")
        return f2, f1

def find_image(folder, fid):
    for ext in (".jpg", ".png", ".jpeg"):
        p = os.path.join(folder, fid + ext)
        if os.path.exists(p):
            return p
    return None

def process_split(split):
    split_src = os.path.join(SRC_DIR, split)
    rgb_in, ir_in = resolve_modal_folders(split_src)
    lbl_in = os.path.join(split_src, "labels")

    rgb_out = os.path.join(OUT_DIR, split, "images")      # YOLOv8 standart taraması için RGB ana dizin
    ir_out = os.path.join(OUT_DIR, split, "images_ir")    # Eşleşmiş IR dizini
    lbl_out = os.path.join(OUT_DIR, split, "labels")

    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(ir_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    count = 0
    for lbl_file in sorted(os.listdir(lbl_in)):
        if not lbl_file.endswith(".txt"):
            continue
        fid = Path(lbl_file).stem

        rgb_path = find_image(rgb_in, fid)
        ir_path = find_image(ir_in, fid)
        if rgb_path is None or ir_path is None:
            continue

        new_lines = []
        with open(os.path.join(lbl_in, lbl_file), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])

                px_xc, px_yc = xc * ORIG_W, yc * ORIG_H
                px_w, px_h = w * ORIG_W, h * ORIG_H

                nx_xc = px_xc - X_MIN
                nx_yc = px_yc - Y_MIN

                x1 = max(0.0, nx_xc - px_w / 2.0)
                y1 = max(0.0, nx_yc - px_h / 2.0)
                x2 = min(float(CROP_W), nx_xc + px_w / 2.0)
                y2 = min(float(CROP_H), nx_yc + px_h / 2.0)

                nw, nh = x2 - x1, y2 - y1
                if nw <= 2.0 or nh <= 2.0:
                    continue

                f_xc = ((x1 + x2) / 2.0) / CROP_W
                f_yc = ((y1 + y2) / 2.0) / CROP_H
                f_w = nw / CROP_W
                f_h = nh / CROP_H
                new_lines.append(f"{cid} {f_xc:.6f} {f_yc:.6f} {f_w:.6f} {f_h:.6f}")

        if not new_lines:
            continue

        rgb_crop = cv2.imread(rgb_path)[Y_MIN:Y_MIN+CROP_H, X_MIN:X_MIN+CROP_W]
        ir_crop = cv2.imread(ir_path)[Y_MIN:Y_MIN+CROP_H, X_MIN:X_MIN+CROP_W]

        cv2.imwrite(os.path.join(rgb_out, f"{fid}.jpg"), enhance_rgb(rgb_crop))
        cv2.imwrite(os.path.join(ir_out, f"{fid}.jpg"), enhance_ir(ir_crop))
        with open(os.path.join(lbl_out, lbl_file), "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(new_lines) + "\n")
        count += 1

    print(f"{split.upper()} Fusion veri seti hazırlandı: {count} eşleşmiş RGB-IR çifti.")

if __name__ == "__main__":
    for s in ["train", "val", "test"]:
        process_split(s)