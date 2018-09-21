import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from PIL import Image

import time
import argparse

from utils import save_checkpoint, load_checkpoint
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Training process")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'vgg19'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.01')
    parser.add_argument('--hidden_units', type=int, default=784, help='hidden units2')
    parser.add_argument('--epochs', dest='epochs', default='4')
    parser.add_argument('--gpu', action="store_true", default=True)
    return parser.parse_args()

def train_model(model, criterion, optimizer,trainloader,validloader, epochs,gpu):
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()
    
    steps = 0
    training_loss = 0
    print_every = 50 # print out training loss
    for e in range(epochs): 
        model.train() # turn on dropout
    
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            # Depending on model grad is required    
            # inputs = Variable(inputs, requires_grad=True)
            if gpu and cuda:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
        
            optimizer.zero_grad()
        
        
            # Training
            outputs = model.forward(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculation the loss
            loss.backward() # Backward pass, calculation gradients 
            optimizer.step() # Update the weights using the gadients
        
            training_loss += loss.data[0]  
            if steps % print_every == 0:
                # Model in evaluation mode
                model.eval() # turn off dropout
            
                # --------------------------------------------------
                # Start test loop
                accuracy = 0            
                valid_loss = 0
                for jj, (inputs, labels) in enumerate(validloader):
                    inputs = Variable(inputs, requires_grad=False, volatile = True)
                    labels = Variable(labels)
        
                    inputs = inputs.to(device)
                    labels = labels.to(device)
            
                    outputs = model.forward(inputs)
                    loss = criterion(outputs, labels)
                
                    valid_loss += loss.data[0]
                
                    ## Calculating the accuracy 
                    # Model's output is log-softmax, take exponential to get the probabilities
                    ps = torch.exp(outputs).data
                
                    # Class with highest probability is our predicted class, compare with true label
                    # Gives index of the class with highest probability, max(ps) 
                    equality = (labels.data == ps.max(1)[1])
               
                    # Accuracy is number of correct predictions divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
            # End validation loop
            # --------------------------------------------------
            
                print("Epoch: {}/{}   ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(training_loss),
                      "Valid Loss: {:.3f}.. ".format(valid_loss),
                      "Valid Accuracy %: {:.3f}..".format(100*accuracy/len(validloader)))
            
                training_loss = 0
            
                # Model in training mode
                model.train()
                
            
def main():
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    model = models.vgg16(pretrained = True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    from collections import OrderedDict
    if args.arch == "vgg16":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('input', nn.Linear(feature_num, 3136)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(0.5)),
                                  ('hidden layer', nn.Linear(3136, args.hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(0.5)),
                                  ('hidden layer2', nn.Linear(args.hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "vgg19":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(feature_num, 1024)),
                                  ('drop', nn.Dropout(p=0.5))
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(1024, 102)),
                                  ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = optim.SGD(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = train_data.class_to_idx
    gpu = args.gpu
    train_model(model, criterion, optimizer,trainloader,validloader, epochs,gpu)
    model.class_to_idx = class_index
    save_checkpoint(model, optimizer, args, classifier)


if __name__ == "__main__":
    main()