import os
import pickle
import yaml
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.applications.resnet import ResNet50
from keras.src.metrics import FalsePositives, AUC
from keras.src.regularizers import L2
from keras.src.layers import RandomFlip, RandomRotation, RandomZoom
from keras.src.callbacks import EarlyStopping
from keras.src.utils import image_dataset_from_directory
from sklearn.utils import resample, shuffle
import numpy as np


def oversample_dataset(dataset, class_counts):
    # Separate the dataset into features and labels
    def separate_features_labels(dataset):
        images = []
        labels = []
        for image, label in dataset.unbatch():
            images.append(image.numpy())
            labels.append(np.argmax(label.numpy()))  # Convert one-hot to integer
        return np.array(images), np.array(labels)

    images, labels = separate_features_labels(dataset)

    # Separate the images and labels by class
    class_images = {0: [], 1: [], 2: []}
    class_labels = {0: [], 1: [], 2: []}

    for i, label in enumerate(labels):
        class_images[label].append(images[i])
        class_labels[label].append(label)

    max_samples = max(class_counts.values())

    # Resample each class to have the same number of samples as the majority class
    for class_label in class_counts:
        class_images[class_label] = resample(
            class_images[class_label],
            replace=True,
            n_samples=max_samples,
            random_state=123,
        )
        class_labels[class_label] = [class_label] * max_samples

    # Combine the resampled data
    resampled_images = np.concatenate(
        [class_images[0], class_images[1], class_images[2]]
    )
    resampled_labels = np.concatenate(
        [class_labels[0], class_labels[1], class_labels[2]]
    )

    # Shuffle the resampled dataset
    resampled_images, resampled_labels = shuffle(
        resampled_images, resampled_labels, random_state=123
    )

    # Create a new TensorFlow dataset from the resampled data
    resampled_labels = tf.keras.utils.to_categorical(
        resampled_labels, num_classes=len(class_counts)
    )
    resampled_dataset = tf.data.Dataset.from_tensor_slices(
        (resampled_images, resampled_labels)
    )

    # Optional: Batch the dataset
    resampled_dataset = resampled_dataset.batch(32)

    return resampled_dataset


def create_model(height, width, num_classes):
    imported_model = ResNet50(
        include_top=False,
        input_shape=(height, width, 3),
        pooling="avg",
        weights="imagenet",
    )

    for layer in imported_model.layers[:-26]:
        layer.trainable = False

    dnn_model = Sequential(
        [
            imported_model,
            Dense(512, activation="relu", kernel_regularizer=L2(0.01)),
            Dropout(0.5),
            Dense(num_classes, activation="softmax", kernel_regularizer=L2(0.01)),
        ]
    )

    return dnn_model


def create_data_augmentation_pipeline():
    data_augmentation = Sequential(
        [RandomFlip("horizontal"), RandomRotation(0.2), RandomZoom(0.2)]
    )

    return data_augmentation


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_model(train_set, validation_set, config):
    height = config["image_height"]
    width = config["image_width"]
    num_classes = config["num_classes"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    model_save_path = config["model_save_path"]
    class_counts = config["class_counts"]

    dnn_model = create_model(height, width, num_classes)

    dnn_model.compile(
        optimizer=Adam(learning_rate=learning_rate, decay=weight_decay),
        loss="categorical_crossentropy",
        metrics=["accuracy", FalsePositives(), AUC(from_logits=False)],
    )

    dnn_model.summary()

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True,
        start_from_epoch=20,
    )

    oversampled_train_set = oversample_dataset(train_set, class_counts)

    data_augmentation = create_data_augmentation_pipeline()
    augmented_train_set = oversampled_train_set.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    )  # Apply data augmentation

    history = dnn_model.fit(
        augmented_train_set,
        validation_data=validation_set,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
    )

    # Save the model
    os.makedirs(model_save_path, exist_ok=True)
    dnn_model.save(filepath=os.path.join(model_save_path, "model.keras"))
    with open(os.path.join(model_save_path, "history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    return history


if __name__ == "__main__":
    config_file = "configs/oversampling_config.yaml"
    config = load_config(config_file)

    train_set = image_dataset_from_directory(
        config["train_dataset_path"],
        seed=123,
        image_size=(config["image_height"], config["image_width"]),
        batch_size=config["batch_size"],
        label_mode="categorical",
    )

    validation_set = image_dataset_from_directory(
        config["validation_dataset_path"],
        seed=123,
        image_size=(config["image_height"], config["image_width"]),
        batch_size=config["batch_size"],
        label_mode="categorical",
    )

    # Train the model using the loaded configuration
    history = train_model(train_set, validation_set, config)
