import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_labels(label_path, fallback_dataset_dir):
    if os.path.exists(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if labels:
            return labels
    classes = sorted(
        [d for d in os.listdir(fallback_dataset_dir) if os.path.isdir(os.path.join(fallback_dataset_dir, d))]
    )
    return classes


def resolve_labels(model, label_path, dataset_dir):
    labels = load_labels(label_path, dataset_dir)
    # Handle models where output shape is (None, num_classes)
    model_classes = int(model.output_shape[-1])
    if len(labels) == model_classes:
        return labels

    fallback = sorted(
        [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    )
    if len(fallback) == model_classes:
        print(
            f"Label count mismatch (labels={len(labels)} vs model={model_classes}). "
            "Using dataset folder class names instead."
        )
        return fallback

    print(
        f"Label count mismatch (labels={len(labels)}, dataset={len(fallback)}, model={model_classes}). "
        "Using generic class names."
    )
    return [f"class_{i}" for i in range(model_classes)]


def predict_with_unknown(model, labels, img_path, img_size, confidence_threshold, margin_threshold):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    probs = model.predict(img_array, verbose=0)[0]
    top2_idx = np.argsort(probs)[-2:][::-1]
    top1_idx, top2_idx = int(top2_idx[0]), int(top2_idx[1])
    top1_conf = float(probs[top1_idx])
    top2_conf = float(probs[top2_idx])
    margin = top1_conf - top2_conf

    is_unknown = (top1_conf < confidence_threshold) or (margin < margin_threshold)
    label = "unknown" if is_unknown else labels[top1_idx]
    reason = None
    if is_unknown:
        if top1_conf < confidence_threshold:
            reason = f"low_confidence({top1_conf:.4f} < {confidence_threshold:.2f})"
        else:
            reason = f"low_margin({margin:.4f} < {margin_threshold:.2f})"

    return {
        "label": label,
        "top1_label": labels[top1_idx],
        "top1_confidence": top1_conf,
        "top2_label": labels[top2_idx],
        "top2_confidence": top2_conf,
        "margin": margin,
        "reason": reason,
        "probs": probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict plant class with unknown rejection")
    parser.add_argument("--model", default="herbaura_best.keras")
    parser.add_argument("--labels", default="labels.txt")
    parser.add_argument("--dataset_dir", default="dataset/raw")
    parser.add_argument("--image", default=None, help="Path to input image. If omitted, you will be prompted.")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--conf_th", type=float, default=0.60, help="Top-1 confidence threshold")
    parser.add_argument("--margin_th", type=float, default=0.15, help="Top1-Top2 margin threshold")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    labels = resolve_labels(model, args.labels, args.dataset_dir)
    print("Model:", args.model)
    print("Type image path and press Enter. Type 'exit' to quit.")

    while True:
        if args.image:
            img_path = args.image
            args.image = None
        else:
            img_path = input("Enter image path (or 'exit'): ").strip().strip('"').strip("'")

        if not img_path:
            print("Empty path. Try again.")
            continue
        if img_path.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        if not os.path.isfile(img_path):
            print(f"Not a file: {img_path}")
            continue
        ext = os.path.splitext(img_path)[1].lower()
        if ext and ext not in VALID_EXTS:
            print(f"Unsupported extension: {ext}. Supported: {sorted(VALID_EXTS)}")
            continue

        try:
            result = predict_with_unknown(
                model=model,
                labels=labels,
                img_path=img_path,
                img_size=args.img_size,
                confidence_threshold=args.conf_th,
                margin_threshold=args.margin_th,
            )
        except Exception as e:
            print(f"Failed to process image: {img_path}")
            print(f"Error: {e}")
            continue

        print("Image:", img_path)
        print("Prediction:", result["label"])
        print(f"Top1: {result['top1_label']} ({result['top1_confidence']:.4f})")
        print(f"Top2: {result['top2_label']} ({result['top2_confidence']:.4f})")
        print(f"Margin: {result['margin']:.4f}")
        if result["reason"]:
            print("Unknown reason:", result["reason"])
        print("-" * 50)


if __name__ == "__main__":
    main()
