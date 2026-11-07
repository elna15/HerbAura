import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate plant model with detailed confusion analysis")
    parser.add_argument("--model", default="herbaura_best.keras", help="Path to .keras model")
    parser.add_argument("--data_dir", default="dataset/raw", help="Dataset root directory")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split for reporting results (paper setting uses 0.2).",
    )
    parser.add_argument("--top_k", type=int, default=10, help="Top confused class pairs to print")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    print(f"Model loaded: {args.model}\n")

    use_split = 0.0 < args.val_split < 1.0
    if not use_split:
        raise ValueError("--val_split must be between 0 and 1 (exclusive). Example: 0.2")
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=args.val_split if use_split else 0.0,
    )

    val_generator = val_datagen.flow_from_directory(
        args.data_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=args.seed,
    )

    # Dataset size and class-count clarity for paper reporting
    all_classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    full_counts = {
        c: sum(
            1
            for f in os.listdir(os.path.join(args.data_dir, c))
            if os.path.isfile(os.path.join(args.data_dir, c, f))
        )
        for c in all_classes
    }
    val_counts = np.bincount(val_generator.classes, minlength=len(val_generator.class_indices))
    inv_idx = {v: k for k, v in val_generator.class_indices.items()}

    print("\n================ DATASET SUMMARY ================")
    print(f"Data directory: {args.data_dir}")
    print(f"Total classes in directory: {len(all_classes)}")
    print(f"Total images in directory: {sum(full_counts.values())}")
    print(f"Validation split used: {args.val_split}")
    print("Per-class counts (full -> validation subset):")
    for i in range(len(inv_idx)):
        cname = inv_idx[i]
        print(f"  {cname}: {full_counts.get(cname, 0)} -> {int(val_counts[i])}")

    y_true = val_generator.classes
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_indices = val_generator.class_indices
    class_names = [None] * len(class_indices)
    for name, idx in class_indices.items():
        class_names[idx] = name

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    overall_acc = accuracy_score(y_true, y_pred)
    print("\n================ OVERALL METRICS ================")
    print(f"Total samples: {len(y_true)}")
    print(f"Overall accuracy: {overall_acc * 100:.2f}%")

    print("\n================ CLASSIFICATION REPORT ================")
    labels = list(range(len(class_names)))
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    print("================ PER PLANT REPORT ================\n")
    for i, plant in enumerate(class_names):
        total = int(np.sum(cm[i]))
        tp = int(cm[i][i])
        row_acc = (tp / total) * 100 if total > 0 else 0.0

        pred_as_i = int(np.sum(cm[:, i]))
        precision = (tp / pred_as_i) * 100 if pred_as_i > 0 else 0.0
        recall = row_acc
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"Plant: {plant}")
        print(f"Total Images: {total}")
        print(f"Correctly Classified: {tp}")
        print(f"Recall (class accuracy): {recall:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"F1-score: {f1:.2f}%")

        misclassified = []
        for j, other_plant in enumerate(class_names):
            if i != j and cm[i][j] > 0:
                misclassified.append((other_plant, int(cm[i][j])))

        if misclassified:
            misclassified.sort(key=lambda x: x[1], reverse=True)
            print("Misclassified As:")
            for name, cnt in misclassified:
                print(f"   -> {name}: {cnt}")
        else:
            print("No misclassifications.")

        print("\n-------------------------------------------------\n")

    # Top confused pairs
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                pairs.append((int(cm[i][j]), class_names[i], class_names[j]))

    pairs.sort(reverse=True, key=lambda x: x[0])

    print("================ TOP CONFUSED PAIRS ================")
    if not pairs:
        print("No confusions found.")
    else:
        for cnt, src, dst in pairs[: args.top_k]:
            print(f"{src} -> {dst}: {cnt}")

    # Optional matrix dump for debugging
    print("\n================ CONFUSION MATRIX (rows=true, cols=pred) ================")
    print("Classes:", class_names)
    print(cm)


if __name__ == "__main__":
    main()
