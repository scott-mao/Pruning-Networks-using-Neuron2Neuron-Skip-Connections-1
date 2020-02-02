import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


###Downloading the dataset - MNIST
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

trainset_MNIST = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
train_loader_MNIST = torch.utils.data.DataLoader(trainset_MNIST, batch_size=50,
                                          shuffle=True, num_workers=2)

valset_MNIST = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
val_loader_MNIST = torch.utils.data.DataLoader(valset_MNIST, batch_size=50,
                                         shuffle=False, num_workers=2)


###Downloading the dataset - CIFAR 10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=50,
                                          shuffle=True, num_workers=2)

valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(valset, batch_size=50,
                                         shuffle=False, num_workers=2)

print(len(trainset),len(valset))
print(len(train_loader),len(val_loader))

def train1_conv(model1):
    criterion =nn.NLLLoss()
    model1 = model1.cuda()
    optimizer = optim.SGD(model1.parameters(), lr=0.003, momentum=0.9)
    time0 = time.time()
    training_loss = []
    epochs = 50
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        for data, labels in train_loader:
            # Flatten images
            data = data.view(data.shape[0], -1)

            # Training pass
            optimizer.zero_grad()
            data,labels = data.cuda(),labels.cuda()
            output = model1(data.double())
            
            loss = criterion(output.double(), labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss/len(train_loader)}")
        training_loss.append(running_loss/len(train_loader))
    print("\nTraining Time (in minutes) =",(time.time()-time0)/60)
    return training_loss

    
def train_conv(model1):
    criterion =nn.NLLLoss()
    model1 = model1.cuda()
    optimizer = optim.SGD(model1.parameters(), lr=0.003, momentum=0.9)
    time0 = time.time()
    training_loss = []
    epochs = 10
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        for data, labels in train_loader:
            # Flatten images
            #data = data.view(data.shape[0], -1)

            # Training pass
            optimizer.zero_grad()
            data,labels = data.cuda(),labels.cuda()
            output = model1(data)
            
            loss = criterion(output, labels)

            #This is where the model learns by backpropagating
            loss.backward()

            #And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        training_loss.append(running_loss/len(train_loader))
        print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss/len(train_loader)}")
    print("\nTraining Time (in minutes) =",(time.time()-time0)/60)
    return training_loss

def train1(model1):
    minimum = 1000
    criterion =nn.NLLLoss()
    model1 = model1.cuda()
    optimizer = optim.SGD(model1.parameters(), lr=0.003, momentum=0.9)
    time0 = time.time()
    training_loss,validation_loss = [],[]
    epochs = 50
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        for data, labels in train_loader:
            # Flatten images
            data = data.view(data.shape[0], -1)

            # Training pass
            optimizer.zero_grad()
            data,labels = data.cuda(),labels.cuda()
            output = model1(data.double())
            
            loss = criterion(output.double(), labels)

            #This is where the model learns by backpropagating
            loss.backward()

            #And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        training_loss.append(running_loss/len(train_loader))
        ###Validation Loss
        correct = 0
        total = 0
        val_loss = 0
        model1.eval()
        with torch.no_grad():
            for data, targets in val_loader:
                # Flatten images
                data = data.view(data.shape[0], -1)

                # Training pass
                optimizer.zero_grad()
                data,targets = data.cuda(),targets.cuda()
                outputs = model1(data.double())

                test_loss = criterion(outputs.double(), targets)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                val_loss += test_loss.item()
        validation_loss.append(val_loss/len(val_loader))
        if minimum > (val_loss/len(val_loader)):
          minimum = val_loss/len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss/len(train_loader)} Validation loss: {val_loss/len(val_loader)}")
        if np.allclose(val_loss/len(val_loader),minimum,atol=0.03):
          continue
        else:
          break
    print("\nTraining Time (in minutes) =",(time.time()-time0)/60)
    return training_loss,validation_loss
    
def train(model1):
    minimum = 1000
    criterion =nn.NLLLoss()
    model1 = model1.cuda()
    optimizer = optim.SGD(model1.parameters(), lr=0.003, momentum=0.9)
    time0 = time.time()
    training_loss = []
    validation_loss = []
    epochs = 50
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        for data, labels in train_loader:
            # Flatten images
            data = data.view(data.shape[0], -1)

            # Training pass
            optimizer.zero_grad()
            data,labels = data.cuda(),labels.cuda()
            output = model1(data)
            loss = criterion(output, labels)

            #This is where the model learns by backpropagating
            loss.backward()

            #And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        training_loss.append(running_loss/len(train_loader))
        ###Validation Loss
        correct = 0
        total = 0
        val_loss = 0
        model1.eval()
        with torch.no_grad():
            for data, targets in val_loader:
                # Flatten images
                data = data.view(data.shape[0], -1)

                # Training pass
                optimizer.zero_grad()
                data,targets = data.cuda(),targets.cuda()
                outputs = model1(data)

                test_loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                val_loss += test_loss.item()
        validation_loss.append(val_loss/len(val_loader))
        if minimum > (val_loss/len(val_loader)):
          minimum = val_loss/len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Training loss: {running_loss/len(train_loader)} Validation loss: {val_loss/len(val_loader)}")
        if np.allclose(val_loss/len(val_loader),minimum,atol=0.03):
          continue
        else:
          break
    print("\nTraining Time (in minutes) =",(time.time()-time0)/60)
    return training_loss,validation_loss



def validation(model1):
    correct_count, all_count = 0, 0
    model1.eval()
    for images,labels in val_loader:
      for i in range(len(labels)):
        img = images[i].view(1, 3072)
        with torch.no_grad():
            img,labels = img.cuda(),labels.cuda()
            logps = model1(img)

        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu().numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (100*correct_count/all_count))
    return 100*correct_count/all_count
    
def validation1(model1):
    correct_count, all_count = 0, 0
    model1.eval()
    for images,labels in val_loader:
      for i in range(len(labels)):
        img = images[i].view(1, 3072)
        with torch.no_grad():
            img,labels = img.cuda(),labels.cuda()
            logps = model1(img.double())

        ps = torch.exp(logps.double())
        probab = list(ps.double().cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu().numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (100*correct_count/all_count))
    return 100*correct_count/all_count
'''
def validation_conv(model1):
    correct_count, all_count = 0, 0
    model1.eval()
    for images,labels in val_loader:
      for i in range(len(labels)):
        img = images[i]
        with torch.no_grad():
            img,labels = img.cuda(),labels.cuda()
            logps = model1(img,sparsity)

        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu().numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (100*correct_count/all_count))
    return 100*correct_count/all_count
'''
def validation_conv(model):
  correct = 0
  total = 0
  model = model.cuda()
  with torch.no_grad():
    for data in val_loader:
      images, labels = data
      images, labels = images.cuda(),labels.cuda()
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    print("Accuracy: ", 100*correct/total)
    return 100*correct/total

def plots(training_loss,validation_loss):
  fig,ax = plt.subplots()
  ax.plot(training_loss,label='Training_loss')
  ax.plot(validation_loss,label='Validation_loss')
  plt.legend()
  plt.show()
