import os
import json
from PIL import Image
import pathlib
import numpy as np
import math

DATA_PATH = pathlib.Path("../../data")


def adjust_bbox_coordinates(bbox, image_width, image_height):
    """Adjust bounding box coordinates to stay within image bounds."""
    x, y, width, height = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(x + width, image_width)
    y2 = min(y + height, image_height)
    return x1, y1, x2, y2


def crop_and_resize_image(img, bbox):
    """Crop the image using the bounding box coordinates and resize it to (128, 128)."""
    x1, y1, x2, y2 = bbox
    class_img = img.crop((x1, y1, x2, y2))
    class_img = class_img.resize((128, 128))
    return class_img


def process_citypersons_dataset(bbox_path, img_path, output_base_path):
    """Process the CityPersons dataset."""
    # Create output directories if they don't exist
    pedestrian_classes = [
        "ignore",
        "pedestrian",
        "rider",
        "sitting person",
        "person (other)",
        "person group",
    ]
    for cls in pedestrian_classes:
        output_path = os.path.join(output_base_path, cls)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    # Process each JSON file
    for city in os.listdir(bbox_path):
        city_bbox_path = os.path.join(bbox_path, city)
        for file in os.listdir(city_bbox_path):
            if file.endswith(".json"):
                json_file = os.path.join(city_bbox_path, file)
                with open(json_file, "r") as f:
                    data = json.load(f)

                img_file = os.path.join(
                    img_path,
                    city,
                    file.replace("_gtBboxCityPersons.json", "_leftImg8bit.png"),
                )
                img = Image.open(img_file)
                image_width, image_height = img.size

                for obj in data["objects"]:
                    cls = obj["label"]
                    if cls in pedestrian_classes:
                        bbox = obj["bbox"]
                        x1, y1, x2, y2 = adjust_bbox_coordinates(
                            bbox, image_width, image_height
                        )

                        # Crop and resize the image
                        class_img = crop_and_resize_image(img, (x1, y1, x2, y2))

                        # Save the extracted class image
                        output_path = os.path.join(output_base_path, cls)
                        base_name = file.replace(
                            "_gtBboxCityPersons.json", f"_{cls}.png"
                        )
                        output_file = os.path.join(output_path, base_name)
                        class_img.save(output_file)


def main():
    cityscapes_path = DATA_PATH / "raw"
    bbox_path = cityscapes_path / "gtBboxCityPersons"
    img_path = cityscapes_path / "leftImg8bit"
    output_base_path = DATA_PATH / "processed"
    if not os.path.exists(output_base_path / "train"):
        os.makedirs(output_base_path / "train")
    if not os.path.exists(output_base_path / "val"):
        os.makedirs(output_base_path / "val")
    process_citypersons_dataset(
        bbox_path / "train", img_path / "train", output_base_path / "train"
    )
    process_citypersons_dataset(
        bbox_path / "val", img_path / "val", output_base_path / "val"
    )


if __name__ == "__main__":
    main()
