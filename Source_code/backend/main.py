import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from ultralytics import YOLO

app = FastAPI(title="Corn Leaf Disease Classification API")

# Custom layers for Vision Transformer
@tf.keras.utils.register_keras_serializable()
class Patches(tf.keras.layers.Layer):
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
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
VIT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Model', 'YOLO_ViT_best.keras'))
YOLO_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'best_yolo_v17.pt'))
IMAGE_SIZE = (224, 224)
CLASSES = ["blight", "common_rust", "gray_spot", "healthy", "not_corn_leaf"]

# Load model globally
model = None
yolo_model = None

@app.on_event("startup")
async def load_keras_model():
    global model, yolo_model
    try:
        print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO Model loaded successfully.")
        
        print(f"Loading ViT model from {VIT_MODEL_PATH}...")
        model = tf.keras.models.load_model(
            VIT_MODEL_PATH,
            custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder}
        )
        print("ViT Model loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # YOLO Crop Logic
        if yolo_model is not None:
            w, h = image.size
            results = yolo_model.predict(source=image, conf=0.25, verbose=False)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                xyxy = boxes.xyxy[best_idx].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)

                x1 = max(0, int(x1) - pad_x)
                y1 = max(0, int(y1) - pad_y)
                x2 = min(w, int(x2) + pad_x)
                y2 = min(h, int(y2) + pad_y)

                if x2 > x1 and y2 > y1:
                    image = image.crop((x1, y1, x2, y2))
                    
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image, dtype=np.float32)
        # Normalize to [0, 1]
        image_array = image_array / 255.0
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
        
    image_bytes = await file.read()
    image_tensor = preprocess_image(image_bytes)
    
    # Run prediction
    predictions = model.predict(image_tensor)[0]
    
    # Get highest confidence prediction
    predicted_class_index = np.argmax(predictions)
    predicted_class = CLASSES[predicted_class_index]
    confidence = float(predictions[predicted_class_index])
    
    # Probability distribution
    probabilities = {CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))}
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }
