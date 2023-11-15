import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.io import read_image
import torchvision.transforms as Transform
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/chess_seer")

from sklearn.model_selection import KFold

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nThe code is running using device = {device.type}')

#First, we are going to want to load in the dataset and then augment it
#Must implement 3 functions: __init__, __len__, __getitem__
class ChessPiecesDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0]) 
        image = read_image(image_path)
        image = image.type('torch.FloatTensor').to(device)

        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label

#Some items were sampled from https://www.kaggle.com/datasets/anshulmehtakaggl/chess-pieces-detection-images-dataset?resource=download
#Some pictures were taken personally by me

#Directory path
dir = 'src/augmented/'

#Labels path
labelsPath = 'src/augmentedLabels'

#HyperParameters
batch_size = 64
maxEpochs = 20
learningRate = 0.002
k = 4

training_data = ChessPiecesDataset(labelsPath, dir)

#===================================#
# CNN Classification Model !!!!!!!!!!
#===================================#
class ChessSeerNetwork(nn.Module):
    def __init__(self):
        super(ChessSeerNetwork, self).__init__()

        #1. Conv w/ Relu
        #2. pool
        self.conv1 = nn.Conv2d(3, 6, 5, padding='same')

        #3. conv w/ Relu
        #4. pool
        self.conv2 = nn.Conv2d(6, 16, 5, padding='same')

        #5. conv w/ relu
        #6. pool
        self.conv3 = nn.Conv2d(16, 32, 5, padding='same')

        #self.conv4 = nn.Conv2d(32, 50, 5, padding='same')

        #self.conv5 = nn.Conv2d(50, 25, 5, padding='same')

        # Final dim * initialsize/16 * initialsize/16 (16 = 2^numPools)
        self.fc1 = nn.Linear(32 * 28 * 28, 120)
        #self.fc2 = nn.Linear(120, 60)
        self.fc2 = nn.Linear(120, 12)
        #self.fc3 = nn.Linear(60, 12)

    def forward(self, x):
        x = func.max_pool2d(func.relu(self.conv1(x)), kernel_size=2)
        x = func.max_pool2d(func.relu(self.conv2(x)), kernel_size=2)
        x = func.max_pool2d(func.relu(self.conv3(x)), kernel_size=2)
        #x = func.max_pool2d(func.relu(self.conv4(x)), kernel_size=2)
        #x = func.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = func.relu(self.fc1(x))
        #x = func.relu(self.fc2(x))
        #x = self.fc3(x)
        x = self.fc2(x)
        return x

#===================================#
# Implement Training Loop !!!!!!!!!!!
#===================================#

def train_epoch(model, device, dataloader, loss_fn, optimizer, writer, epochNum, foldNum): #writer, epochNum, and foldNum are for writing the scalars
    train_loss, train_correct = 0.0, 0
    model.train()

    index = 0
    n_total_steps = len(dataloader)
    running_loss = 0.0
    running_correct = 0.0

    #n_total_steps = imagesLen * k-1/k / batch_size
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        #Forward Pass
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)

        #Backward Pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item() #* images.size(0)
        running_loss += loss.item()
        _, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
        running_correct += (predictions == labels).sum().item()

        if (index + 1) % 10 == 0:
            writer.add_scalar('Training loss of fold ' + str(foldNum + 1), running_loss / 10, epochNum * n_total_steps + index)
            writer.add_scalar('Accuracy of fold' + str(foldNum + 1), running_correct / 10, epochNum * n_total_steps + index)
            running_loss = 0
            running_correct = 0

        index += 1
    
    return train_loss, train_correct

def valid_epoch(model, device, dataloader, loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item()*images.size(0)
        scores, predictions = torch.max(output.data, 1)
        val_correct += (predictions == labels).sum().item()
    
    return valid_loss, val_correct

# Iterating through trainLoader will give you a batch of features and labels

splits = KFold(n_splits=k, shuffle=True)          #Implement KFOLDS 

criterion = nn.CrossEntropyLoss()

# For fold in KFolds
for fold, (train_index, val_index) in enumerate(splits.split(np.arange(len(training_data)))):
    model = ChessSeerNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    train_loss, train_correct = 0.0, 0

    print(f'\nFold {fold+1}')

    train_sampler = SubsetRandomSampler(train_index)
    test_sampler = SubsetRandomSampler(val_index)
    train_loader = DataLoader(training_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(training_data, batch_size=batch_size, sampler=test_sampler)

    # Write the graph to file for the fold
    exampleFeatures, exampleLabels = next(iter(train_loader))
    writer.add_graph(model, exampleFeatures)

    for epoch in range(maxEpochs):
        train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer, writer, epoch, fold)
        test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        print(f'Epoch {epoch + 1} / {maxEpochs}:')
        print(f'Average Training Loss: {train_loss:.3f}')
        print(f'Average Test Loss: {test_loss:.3f}')
        print(f'Average Training Accuracy: {train_acc:.3f} %')
        print(f'Average Test Accuracy: {test_acc:.3f} %')

writer.close()
