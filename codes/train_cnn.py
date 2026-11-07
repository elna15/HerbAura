import os
import json
import hashlib
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =========================
# Config loading
# =========================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "train_config.json")


def _deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    default_config = {
        "seed": 42,
        "dataset_path": "dataset/raw",
        "verify_classes": ["kizharnelli", "tulsi"],
        "allowed_ext": [".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        "output_dir": ".",
        "img_size": 224,
        "batch_size": 32,
        "phase1_epochs": 20,
        "phase2_epochs": 20,
        "val_split": 0.2,
        "augmentation": {
            "train": {
                "rotation_range": 15,
                "zoom_range": 0.15,
                "width_shift_range": 0.08,
                "height_shift_range": 0.08,
                "shear_range": 0.05,
                "brightness_range": [0.9, 1.1],
                "horizontal_flip": True,
                "fill_mode": "reflect",
            },
            "val": {},
        },
        "class_weight": {"enabled": True},
        "model": {
            "dense_units": 256,
            "dropout1": 0.4,
            "dropout2": 0.35,
            "label_smoothing": 0.05,
        },
        "optimizer": {"lr_phase1": 1e-3, "lr_phase2": 1e-4},
        "fine_tune": {"unfreeze_last_n": 50, "freeze_batchnorm": True},
        "callbacks": {
            "early_stop_patience": 4,
            "reduce_lr_factor": 0.5,
            "reduce_lr_patience": 2,
            "min_lr": 1e-6,
        },
        "artifacts": {
            "best_checkpoint": "herbaura_best.keras",
            "final_model": "herbaura_full_dataset_model.keras",
            "verification_report": "dataset_verification.json",
            "confusion_matrix": "confusion_matrix.csv",
            "unknown_threshold": "unknown_threshold.json",
        },
        "unknown_threshold_percentile": 5,
    }

    if not os.path.exists(path):
        print(f"Config not found at {path}. Using defaults.")
        return default_config

    with open(path, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    return _deep_update(default_config, user_config)


CONFIG = load_config(CONFIG_PATH)

# =========================
# Reproducibility
# =========================
SEED = int(CONFIG["seed"])
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# Dataset path
# =========================
DATASET_PATH = CONFIG["dataset_path"]
OUTPUT_DIR = CONFIG["output_dir"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Using dataset path:", DATASET_PATH)
print("Absolute path check:", os.path.abspath(DATASET_PATH))
print("Classes found:", sorted(os.listdir(DATASET_PATH)))

# =========================
# Dataset verification (focused on new classes)
# =========================
VERIFY_CLASSES = set(c.lower() for c in CONFIG["verify_classes"])
ALLOWED_EXT = set(CONFIG["allowed_ext"])


def _iter_image_files(class_dir):
    for root, _, files in os.walk(class_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in ALLOWED_EXT:
                yield os.path.join(root, name)


def _file_md5(path, chunk_size=1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(base_dir, focus_classes=None):
    focus_classes = set(c.lower() for c in (focus_classes or []))
    class_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    report = {
        "classes": {},
        "corrupt": {},
        "duplicates": {},
        "totals": {"images": 0, "classes": len(class_dirs)},
    }

    for cls in sorted(class_dirs):
        cls_dir = os.path.join(base_dir, cls)
        lower = cls.lower()
        files = list(_iter_image_files(cls_dir))
        report["classes"][cls] = len(files)
        report["totals"]["images"] += len(files)

        if focus_classes and lower not in focus_classes:
            continue

        corrupt = []
        hashes = defaultdict(list)
        for path in files:
            try:
                _ = tf.keras.utils.load_img(path)
            except Exception:
                corrupt.append(path)
                continue
            try:
                h = _file_md5(path)
                hashes[h].append(path)
            except Exception:
                # If hashing fails, still allow training but report it
                corrupt.append(path)

        dupes = [v for v in hashes.values() if len(v) > 1]
        report["corrupt"][cls] = corrupt
        report["duplicates"][cls] = dupes

    return report


verify_report = verify_dataset(DATASET_PATH, VERIFY_CLASSES)
print("\nDataset verification summary:")
print("Total images:", verify_report["totals"]["images"])
print("Class counts:", verify_report["classes"])
for cls in sorted(VERIFY_CLASSES):
    if cls in verify_report["corrupt"]:
        print(f"Corrupt in {cls}:", len(verify_report["corrupt"][cls]))
    if cls in verify_report["duplicates"]:
        print(f"Duplicates in {cls} groups:", len(verify_report["duplicates"][cls]))

verification_path = os.path.join(OUTPUT_DIR, CONFIG["artifacts"]["verification_report"])
with open(verification_path, "w", encoding="utf-8") as f:
    json.dump(verify_report, f, indent=2)
print("Verification report saved to", verification_path)

# =========================
# Settings
# =========================
IMG_SIZE = (int(CONFIG["img_size"]), int(CONFIG["img_size"]))
BATCH_SIZE = int(CONFIG["batch_size"])
PHASE1_EPOCHS = int(CONFIG["phase1_epochs"])
PHASE2_EPOCHS = int(CONFIG["phase2_epochs"])
VAL_SPLIT = float(CONFIG["val_split"])

# =========================
# Data generators
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VAL_SPLIT,
    **CONFIG["augmentation"]["train"],
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VAL_SPLIT,
    **CONFIG["augmentation"]["val"],
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

print("Training samples:", train_data.samples)
print("Validation samples:", val_data.samples)
print("Total detected images:", train_data.samples + val_data.samples)
print("Class indices:", train_data.class_indices)

# =========================
# Class weights for imbalance
# =========================
class_counts = np.bincount(train_data.classes, minlength=train_data.num_classes)
total = np.sum(class_counts)
if CONFIG["class_weight"]["enabled"]:
    class_weight = {
        i: float(total) / (train_data.num_classes * class_counts[i])
        for i in range(train_data.num_classes)
    }
else:
    class_weight = None
print("Class counts (train split):", class_counts.tolist())
print("Class weights:", class_weight)

# =========================
# Model: MobileNetV2 backbone
# =========================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(float(CONFIG["model"]["dropout1"]))(x)
x = Dense(int(CONFIG["model"]["dense_units"]), activation="relu")(x)
x = Dropout(float(CONFIG["model"]["dropout2"]))(x)
outputs = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

loss_fn = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=float(CONFIG["model"]["label_smoothing"])
)

# =========================
# Callbacks
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=int(CONFIG["callbacks"]["early_stop_patience"]),
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=float(CONFIG["callbacks"]["reduce_lr_factor"]),
    patience=int(CONFIG["callbacks"]["reduce_lr_patience"]),
    min_lr=float(CONFIG["callbacks"]["min_lr"]),
    verbose=1
)

best_ckpt_path = os.path.join(OUTPUT_DIR, CONFIG["artifacts"]["best_checkpoint"])
best_ckpt = ModelCheckpoint(
    best_ckpt_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# =========================
# Phase 1: feature extraction
# =========================
model.compile(
    optimizer=Adam(learning_rate=float(CONFIG["optimizer"]["lr_phase1"])),
    loss=loss_fn,
    metrics=["accuracy"]
)

print("Starting Phase 1 (frozen backbone)...")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=PHASE1_EPOCHS,
    callbacks=[early_stop, reduce_lr, best_ckpt],
    class_weight=class_weight
)

# =========================
# Phase 2: fine-tune top layers
# =========================
base_model.trainable = True
unfreeze_last_n = int(CONFIG["fine_tune"]["unfreeze_last_n"])
for layer in base_model.layers[:-unfreeze_last_n]:
    layer.trainable = False
for layer in base_model.layers[-unfreeze_last_n:]:
    if CONFIG["fine_tune"]["freeze_batchnorm"] and isinstance(layer, BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=float(CONFIG["optimizer"]["lr_phase2"])),
    loss=loss_fn,
    metrics=["accuracy"]
)

print("Starting Phase 2 (fine-tuning top backbone layers)...")
initial_epoch = len(history1.history.get("loss", []))
model.fit(
    train_data,
    validation_data=val_data,
    initial_epoch=initial_epoch,
    epochs=PHASE1_EPOCHS + PHASE2_EPOCHS,
    callbacks=[early_stop, reduce_lr, best_ckpt],
    class_weight=class_weight
)

# =========================
# Save final model
# =========================
final_model_path = os.path.join(OUTPUT_DIR, CONFIG["artifacts"]["final_model"])
model.save(final_model_path)
print("Training complete. Saved:", final_model_path, "and", best_ckpt_path)

# =========================
# Validation evaluation + unknown threshold
# =========================
val_data.reset()
preds = model.predict(val_data, verbose=1)
y_true = val_data.classes
y_pred = np.argmax(preds, axis=1)
max_probs = np.max(preds, axis=1)

# Confusion matrix (CSV for easy inspection)
cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=train_data.num_classes).numpy()
labels = [""] * train_data.num_classes
for name, idx in train_data.class_indices.items():
    labels[idx] = name

cm_path = os.path.join(OUTPUT_DIR, CONFIG["artifacts"]["confusion_matrix"])
with open(cm_path, "w", encoding="utf-8") as f:
    f.write("," + ",".join(labels) + "\n")
    for i, row in enumerate(cm):
        f.write(labels[i] + "," + ",".join(str(x) for x in row) + "\n")
print(f"Confusion matrix saved to {cm_path}")

# Top confusions summary
confusions = []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i == j:
            continue
        if cm[i, j] > 0:
            confusions.append((int(cm[i, j]), labels[i], labels[j]))
confusions.sort(reverse=True)
top_confusions = confusions[:10]
if top_confusions:
    print("Top confusions (count, true, predicted):")
    for count, true_lbl, pred_lbl in top_confusions:
        print(count, true_lbl, "->", pred_lbl)

# Unknown detection threshold: 5th percentile of correct predictions
correct_mask = y_pred == y_true
correct_probs = max_probs[correct_mask]
if correct_probs.size == 0:
    unknown_threshold = 0.7
else:
    unknown_threshold = float(np.percentile(correct_probs, CONFIG["unknown_threshold_percentile"]))

threshold_info = {
    "unknown_threshold": unknown_threshold,
    "percentile": CONFIG["unknown_threshold_percentile"],
    "note": "If max softmax probability is below this, treat as unknown.",
}
threshold_path = os.path.join(OUTPUT_DIR, CONFIG["artifacts"]["unknown_threshold"])
with open(threshold_path, "w", encoding="utf-8") as f:
    json.dump(threshold_info, f, indent=2)
print("Unknown threshold saved to", threshold_path + ":", unknown_threshold)
