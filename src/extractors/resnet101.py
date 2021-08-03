
import os
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet import ResNet101, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np


class ResNet101Extractor:
    def __init__(self):

        # Resnet50v2
        base_model = ResNet101(weights='imagenet')
        self.model = Model(inputs=base_model.input,
                           outputs=base_model.get_layer('avg_pool').output)

    def extract(self, img) -> np.ndarray:
        '''Extract features from image and return them as an array.

        Args:
            img : Image 
            Object from ```PIL.Image.open(path)``` or ```keras.preprocessing.image.load_img(path)```


        '''
        img = img.resize(
            (int(os.environ.get("RESIZE_IMG_SIZE")),
             int(os.environ.get("RESIZE_IMG_SIZE")))
        )

        # Make sure img is color image
        img = img.convert(os.environ.get("IMG_COLOR_MODE_FE"))

        # Apply NasNet Preprocessing
        # To np.array. Height x Width x Channel. dtype=float32
        x = image.img_to_array(img)

        # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )

        return feature / np.linalg.norm(feature)  # Normalize
