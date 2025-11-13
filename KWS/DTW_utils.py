import numpy as np
import pandas as pd


def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)


from PIL import Image, ImageDraw
import numpy as np
from lxml import etree
import regex as re

def extract_word_images(image_nr: int, return_format: str = "PIL"):
    """
    Extract word images from a scanned manuscript page and its corresponding SVG annotation.

    Args:
        image_nr (int): Number of the document to be scanned
    Returns:
        List of (word_image, polygon_points) tuples:
            - word_image: Cropped image of the word.
            - polygon_points: Original polygon coordinates as an (N, 2) NumPy array.
            - location: The location from the transcription DDD-LL-WW
    """
    # Load full page image
    image_path = f"KWS\images\{image_nr}.jpg"
    svg_path = f"KWS\locations\{image_nr}.svg"

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Parse SVG
    tree = etree.parse(svg_path)
    root = tree.getroot()
    polygons = root.findall(".//{*}path")

    word_images = []

    transcriptions = pd.read_csv("KWS/transcription.tsv", delimiter="\t")
    # Extract all the lines that correspond to the file
    doc_nrs = transcriptions[transcriptions.iloc[:,0].str.contains(f"{image_nr}")].iloc[:,0]

    i = 0
    for polygon in polygons:

        location = doc_nrs.iloc[i] # Extract location data for that image
        i += 1
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
        img_np[mask_np == 0] = (255, 255, 255)  # background -> white

        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        crop = img_np[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Convert back to PIL if requested
        if return_format.lower() == "pil":
            crop = Image.fromarray(crop)

        word_images.append((crop, points, location))

    return word_images


def find_word_image(word, images):
    """
    Take a word from keywords.tsv and locate an image with that word in it.
    Args:
        - word: a word from the keywords.tsv file
        - images: the result from the function extract_word_images 
    Returns: An image of that word
    """
    transcriptions = pd.read_csv("KWS/transcription.tsv", delimiter="\t")

    location = transcriptions[transcriptions.iloc[:, 1] == word].iloc[:,0]

    # For now take the first match
    first_loc = location.iloc[0]
    for _, (word_img, poly, loc) in enumerate(images):
        if loc == first_loc:
            return word_img
        
    print("No match found!")
    return None




if __name__ == "__main__":
    words = extract_word_images(272)

    """ # Display or save a few
    for i, (word_img, poly, loc) in enumerate(words[:3]):
        print(loc)
        word_img.show()  # or word_img.save(f"word_{i}.png")  """
    
    img = find_word_image("c-a-r-e-f-u-l", words)
    img.show()

