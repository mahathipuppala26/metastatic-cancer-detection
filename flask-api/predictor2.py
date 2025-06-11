import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

custom_objects = {
    "PatchExtractor": PatchExtractor,
    "PatchEncoder": PatchEncoder,
}

model1 = load_model("vit_model.h5", custom_objects=custom_objects, compile=False)
model2 = load_model("cnn_model.keras", compile=False)

def preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    image = image.resize(target_size)
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image: Image.Image) -> dict:
    vit_img = preprocess_image(image, (128, 128))
    cnn_img = preprocess_image(image, (96, 96))

    vit_pred = model1.predict(vit_img)[0][0]
    cnn_pred = model2.predict(cnn_img)[0][0]

    vit_label = "Cancerous" if vit_pred > 0.5 else "Non Cancerous"
    cnn_label = "Cancerous" if cnn_pred > 0.5 else "Non Cancerous"

    return {
        "vit_model_prediction": vit_label,
        "cnn_model_prediction": cnn_label,
    }
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

custom_objects = {
    "PatchExtractor": PatchExtractor,
    "PatchEncoder": PatchEncoder,
}

model1 = load_model("vit_model.h5", custom_objects=custom_objects, compile=False)
model2 = load_model("cnn_model.keras", compile=False)

def preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    image = image.resize(target_size)
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(image: Image.Image) -> dict:
    vit_img = preprocess_image(image, (128, 128))
    cnn_img = preprocess_image(image, (96, 96))

    vit_pred = model1.predict(vit_img)[0][0]
    cnn_pred = model2.predict(cnn_img)[0][0]

    vit_label = "Cancerous" if vit_pred > 0.5 else "Non Cancerous"
    cnn_label = "Cancerous" if cnn_pred > 0.5 else "Non Cancerous"

    return {
        "vit_model_prediction": vit_label,
        "cnn_model_prediction": cnn_label,
    }
