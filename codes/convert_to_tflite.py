import argparse
import os
import random
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Keras model to TFLite with optional parity verification."
    )
    parser.add_argument(
        "--model",
        default="herbaura_best.keras",
        help="Path to the source .keras model",
    )
    parser.add_argument(
        "--output",
        default="herbaura_best_float.tflite",
        help="Path to the output .tflite file",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Input image size used by the model",
    )
    parser.add_argument(
        "--dataset_dir",
        default="dataset/raw",
        help="Dataset directory used to sample verification images",
    )
    parser.add_argument(
        "--verify_samples",
        type=int,
        default=30,
        help="How many dataset images to use for Keras-vs-TFLite verification",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible verification sampling",
    )
    return parser.parse_args()


def load_image_tensor(img_path, img_size):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0).astype(np.float32)


def collect_dataset_images(dataset_dir):
    all_paths = []
    for class_name in sorted(os.listdir(dataset_dir)):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for name in sorted(os.listdir(class_dir)):
            path = os.path.join(class_dir, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in VALID_EXTS:
                all_paths.append(path)
    return all_paths


def verify_parity(keras_model, tflite_path, dataset_dir, img_size, verify_samples, seed):
    image_paths = collect_dataset_images(dataset_dir)
    if not image_paths:
        print("No verification images found. Skipping parity check.")
        return

    rng = random.Random(seed)
    sample_count = min(verify_samples, len(image_paths))
    sample_paths = rng.sample(image_paths, sample_count)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    max_abs_diff = 0.0
    max_mean_diff = 0.0
    mismatched_top1 = 0

    for path in sample_paths:
        x = load_image_tensor(path, img_size)
        keras_pred = keras_model.predict(x, verbose=0)[0]

        interpreter.set_tensor(input_details["index"], x.astype(input_details["dtype"]))
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_details["index"])[0]

        abs_diff = np.abs(keras_pred - tflite_pred)
        max_abs_diff = max(max_abs_diff, float(np.max(abs_diff)))
        max_mean_diff = max(max_mean_diff, float(np.mean(abs_diff)))

        if int(np.argmax(keras_pred)) != int(np.argmax(tflite_pred)):
            mismatched_top1 += 1

    print("\nVerification summary:")
    print(f"Checked samples: {sample_count}")
    print(f"Top-1 prediction mismatches: {mismatched_top1}")
    print(f"Maximum absolute probability difference: {max_abs_diff:.8f}")
    print(f"Maximum mean probability difference: {max_mean_diff:.8f}")


def main():
    args = parse_args()

    print(f"Loading Keras model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    export_dir = os.path.join(
        os.getcwd(),
        "_tmp_tflite_export",
    )
    shutil.rmtree(export_dir, ignore_errors=True)
    os.makedirs(export_dir, exist_ok=True)
    try:
        print(f"Exporting SavedModel to temporary directory: {export_dir}")
        model.export(export_dir)

        print("Converting to float32 TFLite with no quantization...")
        converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
        converter.optimizations = []
        tflite_model = converter.convert()

        with open(args.output, "wb") as f:
            f.write(tflite_model)
        print(f"Saved TFLite model: {args.output}")

        # Float32 conversion is the closest practical match to the original Keras model.
        verify_parity(
            keras_model=model,
            tflite_path=args.output,
            dataset_dir=args.dataset_dir,
            img_size=args.img_size,
            verify_samples=args.verify_samples,
            seed=args.seed,
        )
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
