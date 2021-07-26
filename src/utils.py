import base64
import io
from PIL import Image


def base64toImage(image_string):
    msg = base64.b64decode(image_string)
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    return img
