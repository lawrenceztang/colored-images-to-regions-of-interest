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
from tqdm import trange
import pandas as pd
import matplotlib.pyplot as plt

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

class UnderwaterDataset(Dataset):

    def __init__(self, root_dir, size, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.size = size
        self.data = []
        for i in trange(size):
            image = Image.open(self.root_dir + "/HAW_2016_48_RAW-" + str(i) + "-input.png")
            imageTarget = Image.open(self.root_dir + "/HAW_2016_48_RAW-" + str(i) + "-target.png")  # type: Image
            pixelsInterest = 0
            width, height = imageTarget.size
            for x in range(width):
                for y in range(height):
                    if sum(imageTarget.getpixel((x, y))) != 0:
                        pixelsInterest += 1

            interest = pixelsInterest / (height * width)

            if self.transform:
                image = self.transform(image)

            self.data.append((image, torch.FloatTensor([interest, 1 - interest])))


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def train_model(model, dataloaders, criterion, optimizer, dict, time_elapsed, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999999
    training_size = dataloaders["train"].dataset.size

    dict_str = str(training_size) + " Training Examples"
    dict[dict_str] = [float("NAN") for t in range(len(time_elapsed))]

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

            epoch_loss = running_loss / (training_size)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                if(epoch != 0):
                    time_elapsed.append(training_size * epoch)
                    dict[dict_str].append(epoch_loss)
                    for key in dict:
                        if key != "time_elapsed" and key != dict_str:
                            dict[key].append(float("NAN"))
        print()

    time_total = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_total // 60, time_total % 60))
    print('Best val loss: {:f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, dict, time_elapsed

def test(model, dataloader):
    running_loss = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)


        outputs = model(inputs)
        outputs = functional.softmax(outputs)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        # statistics
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / (len(dataloader.dataset))
    return epoch_loss

def demo(model, dataloaders):
    model.eval()
    for inputs, labels in dataloaders["val"]:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        outputs = functional.softmax(outputs)
        print("predicted: " + str(outputs))
        print("ground truth: " + str(labels))

def get_model():
    model = models.resnet18(pretrained=True)
    set_parameter_requires_grad(model, feature_extract)
    model.fc = nn.Linear(512, num_classes)
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
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    optimizer = optim.Adam(params_to_update, lr=0.002)
    return model, optimizer

num_classes = 2
feature_extract = True
batch_size = 10
num_epochs = 25
input_size = 224
criterion = nn.MSELoss()

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

training_sizes = [5 * 2**x for x in range(10)]

dataset_num = 0

while(True):
    try:
        train_dataset = UnderwaterDataset("dataset" + str(dataset_num) + "/train", 6000, transform=data_transforms["train"])
        val_dataset = UnderwaterDataset("dataset" + str(dataset_num) + "/val", 1000, transform=data_transforms["val"])
        test_dataset = UnderwaterDataset("dataset" + str(dataset_num) + "/test", 1, transform=data_transforms["val"])
    except:
        break
    dataset_num += 1


    image_datasets = {"train": train_dataset,
                      "val": val_dataset}

    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=1) for x in ['train', 'val']}
    dataloader_test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Send the model to GPU
    device = torch.device("cuda:0")

    dict = {}
    time_elapsed = []

    for g in trange(len(training_sizes)):
        train_dataset.size = training_sizes[g]
        model, optimizer = get_model()
        # Train and evaluate
        model_ft, dict, time_elapsed = train_model(model, dataloaders_dict, criterion, optimizer, dict, time_elapsed, num_epochs=num_epochs)
        torch.save(model.state_dict(), "model-" + str(dataset_num) + "-" + str(training_sizes[g]) + ".pt")

    import pickle

    file_h = open("dict-" + str(dataset_num) + ".obj", "w+b")
    pickle.dump(dict, file_h)
    file_h = open("time_elapsed.obj", "w+b")
    pickle.dump(time_elapsed, file_h)
    data_frame = pd.DataFrame(dict, index=time_elapsed)
    plt.figure()
    data_frame.plot()
    plt.savefig(str(dataset_num) + "-plot")
