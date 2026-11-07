import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# SETTINGS
# =========================
IMG_SIZE = (224, 224)
DATASET_PATH = "dataset/raw"
TFLITE_MODEL_PATH = "herbaura_best_float.tflite"

# =========================
# LOAD TFLITE MODEL
# =========================
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# VALIDATION DATA
# =========================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print(f"Found {val_data.samples} images belonging to {val_data.num_classes} classes.")

# =========================
# PREPARE CLASS MAPPINGS
# =========================
class_indices = val_data.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

per_class_total = {k: 0 for k in class_indices.keys()}
per_class_correct = {k: 0 for k in class_indices.keys()}
misclassified_map = {k: {} for k in class_indices.keys()}

# =========================
# EVALUATION LOOP
# =========================
for i in range(len(val_data)):
    image, label = val_data[i]

    # Handle quantized models automatically
    input_dtype = input_details[0]['dtype']

    if input_dtype == np.uint8:
        image = (image * 255).astype(np.uint8)
    elif input_dtype == np.int8:
        image = (image / np.max(np.abs(image))).astype(np.int8)
    else:
        image = image.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(output)
    true_class = np.argmax(label)

    true_name = index_to_class[true_class]
    pred_name = index_to_class[predicted_class]

    per_class_total[true_name] += 1

    if predicted_class == true_class:
        per_class_correct[true_name] += 1
    else:
        if pred_name not in misclassified_map[true_name]:
            misclassified_map[true_name][pred_name] = 0
        misclassified_map[true_name][pred_name] += 1

# =========================
# PRINT REPORT
# =========================
print("\n================ PER PLANT REPORT ================")

for plant in per_class_total:
    total_images = per_class_total[plant]
    correct_images = per_class_correct[plant]
    accuracy = (correct_images / total_images) * 100 if total_images > 0 else 0

    print(f"\nPlant: {plant}")
    print(f"Total Images: {total_images}")
    print(f"Correctly Classified: {correct_images}")
    print(f"Accuracy: {round(accuracy, 2)}%")

    if misclassified_map[plant]:
        print("Misclassified As:")
        for wrong_class, count in misclassified_map[plant].items():
            print(f"   → {wrong_class}: {count}")
    else:
        print("No misclassifications.")

    print("\n-------------------------------------------------")
