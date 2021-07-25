
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np

class FeatureExtractor:
    def __init__(self):

        # Resnet50
        base_model = ResNet50(weights='imagenet')
        self.model = Model(inputs=base_model.input,
                           outputs=base_model.get_layer('avg_pool').output)

        # load the trained model from disk
        # print("[INFO] loading model...")
        # base_model = load_model(config.MODEL_PATH)
        # print(base_model.summary())
        # self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('dense_1').output)

        # NasNetLarge
        # base_model = NASNetLarge(weights='imagenet')
        # self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)

    def extract(self, img):  # img is from PIL.Image.open(path) or keras.preprocessing.image.load_img(path)
        img = img.resize((224, 224))
        img = img.convert('RGB')  # Make sure img is color

        # Apply NasNet Preprocessing
        # To np.array. Height x Width x Channel. dtype=float32
        x = image.img_to_array(img)
        # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Subtracting avg values for each pixel

        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )

        return feature / np.linalg.norm(feature)  # Normalize
