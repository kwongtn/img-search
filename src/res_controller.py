# To be merged and removed

from feature_extractor import FeatureExtractor
import numpy as np
import base64
import io
from PIL import Image

# features = []
# img_paths = []
# for feature_path in glob.glob("static/feature/*"):
#     features.append(pickle.load(open(feature_path, 'rb')))
#     img_paths.append('static/img/' + os.path.splitext(os.path.basename(feature_path))[0] + '.jpg')


def base64toImage(image_string):
    content = image_string.split(';')[1]
    image_encoded = content.split(',')[1]
    msg = base64.b64decode(image_encoded)
    buf = io.BytesIO(msg)
    img = Image.open(buf)
    return img


def inspection(fe: FeatureExtractor, img):
    query = fe.extract(img)

    # Do search by using distance
    dists = np.linalg.norm(features - query, axis=1)

    ids_fully = np.argsort(dists)[:5]
    ids_partially = np.argsort(dists)[:20]
    ids_possibility = np.argsort(dists)[:500]

    fullMatch = [(dists[id], img_paths[id]) for id in ids_fully]
    partialMatch = [(dists[id], img_paths[id]) for id in ids_partially]
    possibleMatch = [(dists[id], img_paths[id]) for id in ids_possibility]
