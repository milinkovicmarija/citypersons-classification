import os
import numpy as np
import pandas as pd
import yaml
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from keras.src.saving import load_model
from keras.src.utils import image_dataset_from_directory, plot_model


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate_model(validation_set, model_save_path, class_labels):
    # Load trained model
    model = load_model(os.path.join(model_save_path, "model.keras"))

    # Define a function to extract true labels
    def extract_labels(images, labels):
        return np.argmax(labels.numpy(), axis=1)

    # Extract true labels and images from the validation set
    y_true = []
    validation_images = []
    for images, labels in validation_set:
        y_true.extend(extract_labels(images, labels))
        validation_images.extend(images.numpy())
    y_true = np.array(y_true)
    validation_images = np.array(validation_images)

    # Generate predictions
    y_pred = np.argmax(model.predict(validation_set), axis=-1)

    # Compute evaluation metrics
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True
    )

    # Save evaluation results
    save_evaluation_results(model_save_path, cm, class_report, class_labels)

    # Plot and save accuracy and loss curves
    plot_accuracy_loss_curves(model_save_path)

    # Visualize and save model architecture
    plot_model_architecture(model, model_save_path)

    # Show and save incorrectly classified images
    show_incorrectly_classified_images(
        validation_images, y_true, y_pred, class_labels, model_save_path
    )


def save_evaluation_results(
    model_save_path, confusion_matrix, class_report, class_labels
):
    # Create directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)

    # Save absolute confusion matrix to CSV
    cm_df = pd.DataFrame(confusion_matrix, index=class_labels, columns=class_labels)
    cm_df.to_csv(os.path.join(model_save_path, "confusion_matrix_absolute.csv"))

    # Compute and save relative confusion matrix to CSV
    cm_normalized = confusion_matrix.astype("float") / confusion_matrix.sum(
        axis=1, keepdims=True
    )
    cm_normalized_df = pd.DataFrame(
        cm_normalized, index=class_labels, columns=class_labels
    )
    cm_normalized_df.to_csv(
        os.path.join(model_save_path, "confusion_matrix_relative.csv")
    )

    # Generate and save classification report
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv(os.path.join(model_save_path, "classification_report.csv"))

    # Calculate TP, FP, FN, TN for each class
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    tn = confusion_matrix.sum() - (tp + fp + fn)

    # Calculate rates
    tp_rate = tp / (tp + fn)
    fp_rate = fp / (fp + tn)
    tn_rate = tn / (tn + fp)
    fn_rate = fn / (fn + tp)

    # Save rates to CSV
    rates_df = pd.DataFrame(
        {
            "Class": class_labels,
            "TP Rate": tp_rate,
            "FP Rate": fp_rate,
            "TN Rate": tn_rate,
            "FN Rate": fn_rate,
        }
    )
    rates_df.to_csv(os.path.join(model_save_path, "rates.csv"))


def plot_accuracy_loss_curves(model_save_path):
    # Load model history
    with open(os.path.join(model_save_path, "history.pkl"), "rb") as f:
        history = pickle.load(f)

    # Plot accuracy and loss curves
    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(accuracy) + 1)

    # Plot training and validation accuracy
    plt.plot(epochs, accuracy, "r-", label="Training accuracy")
    plt.plot(epochs, val_accuracy, "b-", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(model_save_path, "accuracy_curve.png"))
    plt.clf()

    # Plot training and validation loss
    plt.plot(epochs, loss, "r-", label="Training loss")
    plt.plot(epochs, val_loss, "b-", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(model_save_path, "loss_curve.png"))
    plt.clf()


def plot_model_architecture(model, model_save_path):
    # Visualize the model architecture
    plot_model(
        model,
        to_file=os.path.join(model_save_path, "model_architecture.png"),
        show_shapes=True,
        show_layer_names=True,
    )


def show_incorrectly_classified_images(
    images, y_true, y_pred, class_labels, model_save_path
):
    incorrect_indices = np.where(y_true != y_pred)[0]
    os.makedirs(os.path.join(model_save_path, "incorrectly_classified"), exist_ok=True)

    for idx in incorrect_indices:
        image = images[idx]
        true_label = class_labels[y_true[idx]]
        pred_label = class_labels[y_pred[idx]]

        plt.imshow(image.astype("uint8"))
        plt.title(f"True: {true_label}, Pred: {pred_label}")
        plt.axis("off")
        plt.savefig(
            os.path.join(model_save_path, "incorrectly_classified", f"{idx}.png")
        )
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--save-path", type=str, required=True, help="Path to the saved model"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    validation_set = image_dataset_from_directory(
        config["validation_dataset_path"],
        seed=123,
        image_size=(config["image_height"], config["image_width"]),
        batch_size=config["batch_size"],
        label_mode="categorical",
    )

    # Evaluate the model
    evaluate_model(validation_set, args.save_path, config["class_labels"])
