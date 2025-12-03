import cv2
import numpy as np
import os
import glob

DATASET = os.path.join("dataset", "Gravity Falls")
OUT_IMG = "pieces"
os.makedirs(OUT_IMG, exist_ok=True)

def get_grid(folder_name):
    name = folder_name.lower()
    if "2x2" in name:
        return 2
    if "4x4" in name:
        return 4
    if "8x8" in name:
        return 8
    return None

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh = clahe.apply(blur)

    sharp_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharp = cv2.filter2D(enh, -1, sharp_kernel)

    return sharp

for folder in os.listdir(DATASET):
    folder_path = os.path.join(DATASET, folder)
    if not os.path.isdir(folder_path):
        continue

    grid = get_grid(folder)
    if grid is None:
        continue

    images = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        images += glob.glob(os.path.join(folder_path, ext))

    grid_folder = os.path.join(OUT_IMG, f"{grid}x{grid}")
    os.makedirs(grid_folder, exist_ok=True)

    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue

        base = os.path.splitext(os.path.basename(path))[0]

        image_out_folder = os.path.join(grid_folder, base)
        os.makedirs(image_out_folder, exist_ok=True)

        _ = preprocess(img)

        h, w = img.shape[:2]
        tile_h = h // grid
        tile_w = w // grid

        piece_id = 1
        for r in range(grid):
            for c in range(grid):
                y1 = r * tile_h
                y2 = (r + 1) * tile_h
                x1 = c * tile_w
                x2 = (c + 1) * tile_w

                piece = img[y1:y2, x1:x2]

                out_path = os.path.join(
                    image_out_folder,
                    f"{base}_piece_{piece_id}.png"
                )
                cv2.imwrite(out_path, piece)

                piece_id += 1