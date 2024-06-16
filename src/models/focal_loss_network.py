import os
import pickle
import yaml
import tensorflow as tf
from keras import Sequential
from keras.src.layers import (
    Dense,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomBrightness,
    RandomContrast,
)
from keras.src.optimizers import Adam
from keras.src.applications.resnet import ResNet50
from keras.src.metrics import FalsePositives, AUC
from keras.src.regularizers import L2
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.utils import image_dataset_from_directory
from keras.src.losses import CategoricalFocalCrossentropy
import keras_tuner as kt


def create_data_augmentation_pipeline():
    data_augmentation = Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            RandomZoom(0.1),
            RandomBrightness(0.1),
            RandomContrast(0.1),
        ]
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
    alpha = config["alpha"]

    data_augmentation = create_data_augmentation_pipeline()
    train_set = train_set.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    )  # Apply data augmentation

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        start_from_epoch=20,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    def model_builder(hp):
        gamma = hp.Choice("gamma", values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])

        imported_model = ResNet50(
            include_top=False,
            input_shape=(height, width, 3),
            pooling="avg",
            weights="imagenet",
        )

        for layer in imported_model.layers[:-40]:
            layer.trainable = True

        dnn_model = Sequential(
            [
                imported_model,
                Dense(512, activation="relu", kernel_regularizer=L2(0.01)),
                Dropout(0.2),
                Dense(num_classes, activation="softmax", kernel_regularizer=L2(0.01)),
            ]
        )

        dnn_model.compile(
            optimizer=Adam(learning_rate=learning_rate, weight_decay=weight_decay),
            loss=CategoricalFocalCrossentropy(gamma=gamma, alpha=alpha),
            metrics=["accuracy", FalsePositives(), AUC(from_logits=False)],
        )

        return dnn_model

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_loss",
        max_trials=20,
        executions_per_trial=1,
        directory="ktuner",
        project_name="focal_loss_gamma_search",
    )

    tuner.search(
        train_set,
        validation_data=validation_set,
        epochs=10,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best gamma: {best_hps.get('gamma')}")

    dnn_model = tuner.hypermodel.build(best_hps)
    history = dnn_model.fit(
        train_set,
        validation_data=validation_set,
        epochs=num_epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
    )

    os.makedirs(model_save_path, exist_ok=True)
    dnn_model.save(filepath=os.path.join(model_save_path, "model.keras"))
    with open(os.path.join(model_save_path, "history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    return history


if __name__ == "__main__":
    config_file = "configs/focal_loss_config.yaml"
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
