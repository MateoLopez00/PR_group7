import numpy as np


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


from PIL import Image, ImageDraw
import numpy as np
from lxml import etree
import regex as re

def extract_word_images(image_path: str, svg_path: str, return_format: str = "PIL"):
    """
    Extract word images from a scanned manuscript page and its corresponding SVG annotation.

    Args:
        image_path (str): Path to the full page image (.png or .jpg).
        svg_path (str): Path to the corresponding SVG file with <polygon> word boundaries.
        return_format (str): 'PIL' to return PIL Images, 'numpy' to return NumPy arrays.

    Returns:
        List of (word_image, polygon_points) tuples:
            - word_image: Cropped image of the word.
            - polygon_points: Original polygon coordinates as an (N, 2) NumPy array.
    """
    # Load full page image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Parse SVG
    tree = etree.parse(svg_path)
    root = tree.getroot()
    polygons = root.findall(".//{*}path")

    word_images = []

    

    word_images = []

    for polygon in polygons:

        # Extract all numeric coordinate pairs
        d = polygon.attrib["d"]
        #coords = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.?\d+", d)
        coords = re.findall(r"\d*\.?\d+\s+[-+]?\d*\.?\d+", d)

        # Convert matches into float pairs
        points = []
        for pair in coords:
            x, y = map(float, pair.split())
            points.append((x, y))
        points = np.array(points)

        # Create a binary mask for this polygon
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon([tuple(p) for p in points], outline=1, fill=1)
        mask_np = np.array(mask)

        # Apply mask to image and crop bounding box
        img_np = np.array(image)
        img_np[mask_np == 0] = (255, 255, 255)  # background → white

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        crop = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Convert back to PIL if requested
        if return_format.lower() == "pil":
            crop = Image.fromarray(crop)

        word_images.append((crop, points))

    return word_images

if __name__ == "__main__":
    words = extract_word_images("KWS\images/275.jpg", "KWS/locations/275.svg")

    print(f"Extracted {len(words)} word crops")

    # Display or save a few
    for i, (word_img, poly) in enumerate(words[:3]):
        word_img.show()  # or word_img.save(f"word_{i}.png")

