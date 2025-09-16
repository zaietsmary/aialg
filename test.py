import anvil.server

from PIL import Image
import numpy as np
import io
from anvil import BlobMedia
import os

anvil.server.connect("server_N7AC3JL66LLO6HTLONBNHFQ2-O6CK3GJ7NEKRQHCE")

def vector_calculation(img, grid_rows=4, grid_cols=5):
    pixels = np.array(img.convert("1"))
    height, width = pixels.shape
    segment_height = height // grid_rows
    segment_width = width // grid_cols

    vector = [
        np.sum(pixels[i*segment_height:(i+1)*segment_height,
                      j*segment_width:(j+1)*segment_width] == 0)
        for i in range(grid_rows)
        for j in range(grid_cols)
    ]

    total = sum(vector) or 1
    normalized = [v / total for v in vector]
    return vector, normalized

def image_model(folder="img", grid_rows=4, grid_cols=5):
    results = {}
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            img = Image.open(filepath)
            vector, normalized = vector_calculation(img, grid_rows, grid_cols)
            results[filename] = {"absolute": vector, "normalized": normalized}
    return results

models = image_model("img")

def manhattan(v1, v2):
    return np.sum(np.abs(np.array(v1) - np.array(v2)))

def classify_photo(models, photo):
    distances = {name: manhattan(photo["normalized"], v["normalized"]) for name, v in models.items()}
    best_match = min(distances, key=distances.get)
    return best_match, distances

@anvil.server.callable
def process_image(file: BlobMedia, grid_rows=4, grid_cols=5):
    img = Image.open(io.BytesIO(file.get_bytes()))
    vector, normalized = vector_calculation(img, grid_rows, grid_cols)

    out_bytes = io.BytesIO()
    img.save(out_bytes, format="PNG")
    out_bytes.seek(0)

    return vector, normalized, BlobMedia("image/png", out_bytes.read())

@anvil.server.callable
def classify_image(file: BlobMedia, grid_rows=4, grid_cols=5):
    vector, normalized, blob = process_image(file, grid_rows, grid_cols)
    photo = {"absolute": vector, "normalized": normalized}

    best_class, distances = classify_photo(models, photo)
    class_name = os.path.splitext(best_class)[0].rstrip("0123456789")

    return {
        "best_class": class_name,
        "distances": distances,
        "vector": vector,
        "normalized_vector": normalized,
        "image_blob": blob
    }

anvil.server.wait_forever()