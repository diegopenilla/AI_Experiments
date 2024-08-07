"""
This script performs the following tasks:
1. Searches for images using DuckDuckGo and downloads them.
2. Prepares the images for training by resizing and verifying them.
3. Trains a convolutional neural network (CNN) model to classify images as either 'forest' or 'bird'.
4. Searches for a test image using DuckDuckGo, downloads it, and makes a prediction using the trained model.

Functions:
- search_images(term, max_images=30): Searches for images using DuckDuckGo and returns a list of image URLs.
- main(): Main function that orchestrates the image search, download, training, and prediction.

Usage:
Run this script to train a model and make predictions on images of forests and birds.
"""
from duckduckgo_search import DDGS
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    ddgs = DDGS()
    return L(ddgs.images(term, max_results=max_images)).itemgot('image')

searches = 'forest', 'bird'
path = Path('bird_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(5)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(5)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(5)
    resize_images(path/o, max_size=400, dest=path/o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

# train a model
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)


# Search for a test image using DuckDuckGo and save it as 'bird.jpg'
test_image_url = search_images('bird photo', max_images=1)[0]
download_url(test_image_url, 'bird.jpg')

is_bird, _, probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[learn.dls.vocab.o2i['bird']]:.4f}")