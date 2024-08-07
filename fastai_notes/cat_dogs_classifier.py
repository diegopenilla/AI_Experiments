"""
This script trains a cat vs. dog classifier using the Fastai library.

The script performs the following steps:
1. Downloads and extracts the PETS dataset.
2. Defines a function to label images as cat or not based on the filename.
3. Creates an ImageDataLoaders object with the dataset, applying transformations and splitting into training and validation sets.
4. Initializes a vision learner with a ResNet34 architecture and an error rate metric.
5. Fine-tunes the model for one epoch.
6. If run as the main module, prints a success message and performs a prediction on an empty image.

Functions:
    is_cat(x): Returns True if the filename indicates the image is a cat.
"""
from fastai.vision.all import ImageDataLoaders, vision_learner, get_image_files, untar_data, URLs, Resize, resnet34, error_rate
from PIL import Image as PILImage

path = untar_data(URLs.PETS) / 'images'


def is_cat(x): return x[0].isupper()


dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224)
)

learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

if __name__ == '__main__':
    print("Training completed successfully!")
    img = PILImage.create("")
    is_cat, _, probs = learn.predict(img)
    print(f"Is this a cat?: {is_cat}.")
    print(f"Probability it's a cat: {probs[1].item():.6f}")
