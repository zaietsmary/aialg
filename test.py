import anvil.server

from PIL import Image
import numpy as np
import io
from anvil import BlobMedia

anvil.server.connect("server_N7AC3JL66LLO6HTLONBNHFQ2-O6CK3GJ7NEKRQHCE")


@anvil.server.callable
def process_image(file: BlobMedia, grid_rows=4, grid_cols=5):
    img = Image.open(io.BytesIO(file.get_bytes())).convert("1")
    pixels = np.array(img)
    height, width = pixels.shape
    segment_height = height // grid_rows
    segment_width = width // grid_cols

    vector = []
    for i in range(grid_rows):
        for j in range(grid_cols):
            segment = pixels[
                i * segment_height : (i + 1) * segment_height,
                j * segment_width : (j + 1) * segment_width,
            ]
            vector.append(np.sum(segment == 0))
    # за сумою

    total = sum(vector)
    normalized_vector = [v / total for v in vector]
    """
    за модулем:
    n = np.max(vector)
    normalized_vector = [v / n for v in vector]
    """

    out_bytes = io.BytesIO()
    img.save(out_bytes, format="PNG")
    out_bytes.seek(0)
    return vector, normalized_vector, BlobMedia("image/png", out_bytes.read())


anvil.server.wait_forever()
