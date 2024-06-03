import os
import pickle
import yaml
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from keras.src.applications.resnet import ResNet50
from keras.src.metrics import FalsePositives, AUC
from keras.src.utils import image_dataset_from_directory


def create_model(height, width, num_classes):
    imported_model = ResNet50(
        include_top=False,
        input_shape=(height, width, 3),
        pooling="avg",
        weights="imagenet",
    )

    for layer in imported_model.layers[:-26]:
        layer.trainable = False

    dnn_model = Sequential()
    dnn_model.add(imported_model)
    dnn_model.add(Dense(512, activation="relu"))
    dnn_model.add(Dense(num_classes, activation="softmax"))

    return dnn_model


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

    dnn_model = create_model(height, width, num_classes)

    dnn_model.compile(
        optimizer=Adam(learning_rate=learning_rate, weight_decay=weight_decay),
        loss="categorical_crossentropy",
        metrics=["accuracy", FalsePositives(), AUC(from_logits=False)],
    )

    dnn_model.summary()

    history = dnn_model.fit(
        train_set,
        validation_data=validation_set,
        epochs=num_epochs,
        batch_size=batch_size,
    )

    # Save the model
    os.makedirs(model_save_path, exist_ok=True)
    dnn_model.save(filepath=os.path.join(model_save_path, "model.keras"))
    with open(os.path.join(model_save_path, "history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    return history


if __name__ == "__main__":
    config_file = "configs/basic_config.yaml"
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

    # Train the basic model using the loaded configuration
    history = train_model(train_set, validation_set, config)