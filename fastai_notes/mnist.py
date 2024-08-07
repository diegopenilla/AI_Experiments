from fastai.vision.all import *

# Download and extract the MNIST dataset
path = untar_data(URLs.MNIST)

# Create a DataBlock
mnist = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(train_name='training', valid_name='testing'),
    get_y=parent_label,
    batch_tfms=Normalize.from_stats(mean=0.5, std=0.5)
)

# Create DataLoaders
dls = mnist.dataloaders(path, bs=64)

# Visualize some data
dls.show_batch(max_n=9, figsize=(6,6))

# Define a function to modify resnet18 to accept 1-channel input
def create_modified_resnet():
    model = resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

# Create a Learner with the modified ResNet model
learn = Learner(dls, create_modified_resnet(), metrics=accuracy)

# Train the model
learn.fit_one_cycle(1, 0.01)

# Evaluate the model
learn.validate()