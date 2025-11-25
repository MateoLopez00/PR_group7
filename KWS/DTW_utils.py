import numpy as np
import pandas as pd
from skimage.filters import threshold_sauvola

def norm_height(img: np.ndarray, height: int = 80) -> np.ndarray:
    """
    Resize a grayscale image to a fixed height, preserving aspect ratio.
    """
    import cv2

    h, w = img.shape
    new_w = int(w * height / h)
    return cv2.resize(img, (new_w, height))

def binarize(img: np.ndarray) -> np.ndarray:
    """
    Sauvola binarization (image -> {0,1}), as in the exercise sheet.
    """
    T = threshold_sauvola(img, window_size=25)
    return (img < T).astype("uint8")

def preprocess_img(pil_image):
    """
    Full preprocessing pipeline:
      PIL RGB image -> grayscale -> norm_height -> binarize
    Returns a 2D numpy array (H, W) of 0/1.
    """
    import cv2
    from PIL import Image

    if isinstance(pil_image, Image.Image):
        word_img_np = np.array(pil_image)  # PIL → NumPy RGB
    else:
        # Assume already numpy
        word_img_np = pil_image

    word_img_gray = cv2.cvtColor(word_img_np, cv2.COLOR_RGB2GRAY)
    img = norm_height(word_img_gray)
    img = binarize(img)
    return img

def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Extract column-wise features from a binarized word image.
    Returns an array of shape (W, 7) as in the exercise.
    """
    H, W = img.shape
    features = []

    for i in range(W):
        col = img[:, i]
        black = np.where(col == 1)[0]

        # Upper / lower contour
        if len(black) == 0:
            UC, LC = 0, H - 1
        else:
            UC, LC = black[0], black[-1]

        transitions = np.count_nonzero(col[:-1] != col[1:])
        frac_black = np.mean(col)
        between = col[UC:LC + 1] if LC >= UC else []
        frac_between = np.mean(between) if len(between) > 0 else 0

        features.append([
            UC / H,
            LC / H,
            transitions / 10.0,
            frac_black,
            frac_between,
        ])

    # Add gradients of UC/LC
    features = np.array(features, dtype=np.float32)
    grads = np.gradient(features[:, :2], axis=0)
    feats_with_grads = np.concatenate([features, grads], axis=1)  # (W, 7)
    return feats_with_grads

def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Simple Euclidean distance; DTW.py imports this."""
    return float(np.linalg.norm(a - b))

def extract_word_images(image_nr: int, return_format: str = "PIL"):
    """
    Extract word images from a scanned manuscript page and its SVG.
    Returns list of (word_image, polygon_points, location).
    """
    from PIL import Image, ImageDraw
    from lxml import etree
    import regex as re

    image_path = f"images/{image_nr}.jpg"
    svg_path = f"locations/{image_nr}.svg"

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Parse SVG file
    tree = etree.parse(svg_path)
    root = tree.getroot()
    polygons = root.findall(".//{*}path")

    transcriptions = pd.read_csv("transcription.tsv", delimiter="\t")

    # All locations that belong to this document
    doc_nrs = transcriptions[
        transcriptions.iloc[:, 0].str.contains(f"{image_nr}")
    ].iloc[:, 0]

    word_images = []
    i = 0
    for polygon in polygons:
        location = doc_nrs.iloc[i]
        i += 1

        d = polygon.attrib["d"]
        coords = re.findall(r"\d*\.?\d+\s+[-+]?\d*\.?\d+", d)

        points = []
        for pair in coords:
            x, y = map(float, pair.split())
            points.append((x, y))
        points = np.array(points)

        # Build mask and crop
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon([tuple(p) for p in points], outline=1, fill=1)
        mask_np = np.array(mask)

        img_np = np.array(image)
        img_np[mask_np == 0] = (255, 255, 255)

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        crop = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]

        if return_format.lower() == "pil":
            crop = Image.fromarray(crop)

        word_images.append((crop, points, location))

    return word_images

def find_word_image(word: str, images, transcription_path: str = "transcription.tsv"):
    """
    Given a keyword and the list from extract_word_images,
    return one example image of that word (if present).
    """
    transcriptions = pd.read_csv(transcription_path, delimiter="\t")
    loc_series = transcriptions[transcriptions.iloc[:, 1] == word].iloc[:, 0]

    if loc_series.empty:
        print(f"No transcription entry found for word '{word}'")
        return None

    first_loc = loc_series.iloc[0]
    for word_img, poly, loc in images:
        if loc == first_loc:
            return word_img

    print(f"No image match found for word '{word}'")
    return None

if __name__ == "__main__":
    # Tiny self-test (optional)
    words = extract_word_images(272)
    img = find_word_image("c-a-r-e-f-u-l", words)
    if img is not None:
        proc = preprocess_img(img)
        feats = extract_features(proc)
        print("Features shape:", feats.shape)
