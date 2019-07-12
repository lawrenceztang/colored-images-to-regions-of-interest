import torchvision.models as models
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from skimage import io
from torch import optim
import time
import os
import copy
from PIL import Image
import torch.nn.functional as functional

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class UnderwaterDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return 1100

    def __getitem__(self, idx):
        image = Image.open(self.root_dir + "/HAW_2016_48_RAW-" + str(idx) + "-input.png")
        imageTarget = Image.open(self.root_dir + "/HAW_2016_48_RAW-" + str(idx) + "-target.png") # type: Image
        pixelsInterest = 0
        width, height = imageTarget.size
        for x in range(width):
            for y in range(height):
                if sum(imageTarget.getpixel((x, y))) != 0:
                    pixelsInterest += 1

        interest = pixelsInterest / (height * width)

        if self.transform:
            image = self.transform(image)

        sample = (image, torch.FloatTensor([interest, 1 - interest]))

        return sample


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    outputs = functional.softmax(outputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

def demo(model, dataloaders):
    model.eval()
    for inputs, labels in dataloaders["val"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        outputs = functional.softmax(outputs)
        print("predicted: " + str(outputs))
        print("ground truth: " + str(labels))


num_classes = 2
feature_extract = False
batch_size = 1
num_epochs = 50

model = models.resnet18(pretrained=True)
input_size = 224
set_parameter_requires_grad(model, feature_extract)
model.fc = nn.Linear(512, num_classes)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

from torch.utils.data import *
dataset = UnderwaterDataset("/media/lawrence/yogi.ddrive/datasets/100island_coral_reef/HAW_2016_48/dataset", transform=data_transforms["train"])
train_dataset, val_dataset = random_split(dataset, [1000, 100])
val_dataset.dataset.transform = data_transforms["val"]

image_datasets = {"train": train_dataset,
                  "val": val_dataset}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'val']}

# Send the model to GPU
device = torch.device("cuda:0")
model = model.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
optimizer_ft = optim.SGD(params_to_update, lr=0.0000004, momentum=0.9)

# Setup the loss fxn
criterion = nn.MSELoss()

# Train and evaluate
model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
demo(model, dataloaders_dict)
torch.save(model.state_dict(), "model.pt")
import matplotlib.pyplot as plt
plt.plot(hist)
plt.show()
plt.savefig("training_graph.png")

