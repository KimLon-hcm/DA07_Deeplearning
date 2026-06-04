# =========================================
# 1. INSTALL
# =========================================

# !pip install -q ultralytics


# =========================================
# 2. IMPORT
# =========================================

import os
import io
import json
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import confusion_matrix, classification_report

from ultralytics import YOLO

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))


# =========================================
# 3. CONFIG
# =========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Classification dataset 5 class
DATASET_PATH = "/kaggle/input/datasets/kimlonkkk/data10k-v2/split_dataset_backup_10k_v2"



YOLO_MODEL_PATH = "/kaggle/input/datasets/kimlonkkk/yolo17/weights/best.pt"
print("Using YOLO:", YOLO_MODEL_PATH)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

MODEL_NAME = "YOLO_ResNet50_ImageNet_Frozen"
OUT_DIR = f"/kaggle/working/{MODEL_NAME}_results"

CROP_DATASET_DIR = "/kaggle/working/yolo_crop_resnet50_dataset"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CROP_DATASET_DIR, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE

print("Using YOLO:", YOLO_MODEL_PATH)


# =========================================
# 4. CHECK PATHS
# =========================================

print("DATASET_PATH exists:", os.path.exists(DATASET_PATH))
print("YOLO_MODEL_PATH exists:", os.path.exists(YOLO_MODEL_PATH))

print("Dataset folders:", os.listdir(DATASET_PATH))


# =========================================
# 5. LOAD YOLO MODEL
# =========================================

yolo_model = YOLO(YOLO_MODEL_PATH)

print("YOLO classes:", yolo_model.names)


# =========================================
# 6. CROP FUNCTION
# =========================================

def crop_with_yolo(image_path, yolo_model, conf_thres=0.25, img_size=224):
    """
    Detect vùng bệnh bằng YOLO và crop bbox confidence cao nhất.

    Nếu không detect được hoặc box quá nhỏ:
    - dùng ảnh gốc để tránh mất ngữ cảnh.

    Trả về:
    - PIL Image RGB đã resize về img_size.
    """

    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    results = yolo_model.predict(
        source=str(image_path),
        conf=conf_thres,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))

        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()

        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        img_area = w * h

        # Nếu box quá nhỏ thì dùng ảnh gốc
        if box_area / img_area < 0.03:
            return image.resize((img_size, img_size))

        # Padding để giữ thêm ngữ cảnh xung quanh vùng bệnh
        pad_x = int(box_w * 0.5)
        pad_y = int(box_h * 0.5)

        x1 = max(0, int(x1) - pad_x)
        y1 = max(0, int(y1) - pad_y)
        x2 = min(w, int(x2) + pad_x)
        y2 = min(h, int(y2) + pad_y)

        if x2 > x1 and y2 > y1:
            image = image.crop((x1, y1, x2, y2))

    image = image.resize((img_size, img_size))

    return image


# =========================================
# 7. CREATE CROPPED DATASET
# =========================================
# Output:
# /kaggle/working/yolo_crop_resnet50_dataset/
#   train/class_name/*.jpg
#   val/class_name/*.jpg
#   test/class_name/*.jpg

splits = ["train", "val", "test"]

image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

crop_stats = {
    "total": 0,
    "saved": 0,
    "errors": 0
}

for split in splits:
    split_dir = Path(DATASET_PATH) / split

    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        out_class_dir = Path(CROP_DATASET_DIR) / split / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in image_exts
        ]

        print(f"Processing {split}/{class_name}: {len(image_paths)} images")

        for img_path in image_paths:
            crop_stats["total"] += 1

            try:
                cropped = crop_with_yolo(
                    image_path=img_path,
                    yolo_model=yolo_model,
                    conf_thres=0.25,
                    img_size=IMG_SIZE
                )

                out_path = out_class_dir / img_path.name
                cropped.save(out_path)

                crop_stats["saved"] += 1

            except Exception as e:
                crop_stats["errors"] += 1
                print("Error:", img_path, e)

print("Crop stats:", crop_stats)
print("Cropped dataset saved at:", CROP_DATASET_DIR)


# =========================================
# 8. VISUALIZE SOME CROPPED IMAGES
# =========================================

import glob

sample_images = glob.glob(CROP_DATASET_DIR + "/train/*/*")[:12]

plt.figure(figsize=(12, 12))

for i, img_path in enumerate(sample_images):
    img = Image.open(img_path).convert("RGB")

    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(Path(img_path).parent.name)
    plt.axis("off")

plt.tight_layout()
plt.show()


# =========================================
# 9. LOAD CROPPED DATASET
# =========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(CROP_DATASET_DIR, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(CROP_DATASET_DIR, "val"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(CROP_DATASET_DIR, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)

print("Classes:", CLASS_NAMES)
print("Num classes:", NUM_CLASSES)


# =========================================
# 10. PREPROCESS + PREFETCH
# =========================================
# ResNet50 ImageNet cần preprocess_input, KHÔNG dùng Rescaling(1./255)

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)

test_ds = test_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)


# =========================================
# 11. DATA AUGMENTATION NHẸ
# =========================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.05),
        layers.RandomTranslation(0.03, 0.03),
    ],
    name="data_augmentation"
)


# =========================================
# 12. BUILD YOLO + RESNET50 IMAGE-NET FROZEN
# =========================================
# ResNet50 sẽ được train lại trên ảnh đã crop bằng YOLO.

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = data_augmentation(inputs)

# Dataset đã preprocess_input trong tf.data pipeline
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(
    inputs,
    outputs,
    name=MODEL_NAME
)

model.summary()


# =========================================
# 13. COMPILE
# =========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================================
# 14. CALLBACKS
# =========================================

checkpoint = ModelCheckpoint(
    filepath=f"{OUT_DIR}/{MODEL_NAME}_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=1,
    mode="max"
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1,
    mode="min"
)

callbacks = [
    checkpoint,
    earlystop,
    reduce_lr
]


# =========================================
# 15. TRAIN
# =========================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================================
# 16. SAVE HISTORY
# =========================================

history_df = pd.DataFrame(history.history)

history_path = f"{OUT_DIR}/history.csv"
history_df.to_csv(history_path, index=False)

print("Saved:", history_path)
history_df.head()


# =========================================
# 17. LEARNING CURVES
# =========================================

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.title("YOLO + ResNet50 - Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT_DIR}/accuracy_curve.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("YOLO + ResNet50 - Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUT_DIR}/loss_curve.png", dpi=150, bbox_inches="tight")
plt.show()


# =========================================
# 18. EVALUATE TEST SET
# =========================================

test_loss, test_acc = model.evaluate(test_ds)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)


# =========================================
# 19. PREDICT TEST SET
# =========================================

y_true = []

for _, y_batch in test_ds:
    y_true.extend(y_batch.numpy())

y_true = np.array(y_true)

pred_probs = model.predict(test_ds)
y_pred = np.argmax(pred_probs, axis=1)

print("y_true:", y_true.shape)
print("y_pred:", y_pred.shape)


# =========================================
# 20. CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title("YOLO + ResNet50 - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

cm_path = f"{OUT_DIR}/confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.show()

print("Saved:", cm_path)


# =========================================
# 21. CLASSIFICATION REPORT
# =========================================

report_text = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
)

print(report_text)

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()

report_path = f"{OUT_DIR}/classification_report.csv"
report_df.to_csv(report_path)

print("Saved:", report_path)
report_df


# =========================================
# 22. SAVE METRICS
# =========================================

metrics = {
    "model_name": MODEL_NAME,
    "pipeline": "YOLO crop + ResNet50 ImageNet Frozen classifier",
    "backbone": "ResNet50",
    "weights": "imagenet",
    "fine_tune": False,
    "yolo_model_path": YOLO_MODEL_PATH,
    "crop_dataset_dir": CROP_DATASET_DIR,
    "test_loss": float(test_loss),
    "test_accuracy": float(test_acc),
    "classes": CLASS_NAMES,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS
}

metrics_path = f"{OUT_DIR}/metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("Saved:", metrics_path)


# =========================================
# 23. ZIP RESULTS
# =========================================

zip_path = f"/kaggle/working/{MODEL_NAME}_results.zip"

shutil.make_archive(
    zip_path.replace(".zip", ""),
    "zip",
    OUT_DIR
)

print("Saved:", zip_path)


# =========================================
# 24. OPTIONAL - UPLOAD IMAGE FROM COMPUTER AND TEST
# =========================================

from ipywidgets import FileUpload
from IPython.display import display

uploader = FileUpload(
    accept="image/*",
    multiple=False
)

display(uploader)


# =========================================
# 25. PREDICT UPLOADED IMAGE
# =========================================
# Chạy cell này sau khi chọn ảnh ở cell upload.

if len(uploader.value) == 0:
    raise ValueError("Bạn chưa upload ảnh.")

uploaded = uploader.value[0]

img = Image.open(
    io.BytesIO(uploaded["content"])
).convert("RGB")

temp_path = "/kaggle/working/temp_upload_image.jpg"
img.save(temp_path)

cropped = crop_with_yolo(
    image_path=temp_path,
    yolo_model=yolo_model,
    conf_thres=0.25,
    img_size=IMG_SIZE
)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cropped)
plt.title("YOLO crop")
plt.axis("off")

plt.show()

x = np.array(cropped).astype("float32")
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

pred = model.predict(x)[0]

pred_idx = int(np.argmax(pred))
confidence = float(pred[pred_idx])

print("Prediction:", CLASS_NAMES[pred_idx])
print("Confidence:", f"{confidence:.4f}")

print("\nAll probabilities:")
for cls, prob in zip(CLASS_NAMES, pred):
    print(f"{cls:15s}: {prob:.4f}")
