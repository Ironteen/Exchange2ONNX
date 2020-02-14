# !/usr/bin/python
# coding: utf8
# @Time    : 2019-04-03
# @Author  : ZhijunTu
# @Email   : tzj19970116@163.com

import torch 
import torchvision 
import numpy as np 
import os 
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import time
# basic setting 
TRAINING_STEPS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
save_path = './model'
if not os.path.exists(save_path):
    os.mkdir(save_path)
# load data 
Download_mnist = False 
if not (os.path.exists('./data/') and os.listdir('./mnist/')):
    Download_mnist = True
train_data = torchvision.datasets.MNIST(
    root='./data/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=Download_mnist,
)
# train data and test data
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(
    root='./data/',
    train=False,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=Download_mnist,
)
test_loader = Data.DataLoader(dataset=test_data, batch_size=16, shuffle=False)
# create the network
class LeNet_5(nn.Module):
    def __init__(self,output_size,flatten_dim,hidden_1,hidden_2):
        super(LeNet_5,self).__init__()
        self.output_size = output_size
        self.flatten_dim = flatten_dim
        self.hidden1 = hidden_1
        self.hidden2 = hidden_2
        # the 1st layer
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=6,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (6, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (6, 14, 14)
        )
        # the 2nd layer
        self.conv2 = nn.Sequential(         # input shape (6, 14, 14)
            nn.Conv2d(
                in_channels=6,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 7, 7)
        )
        # the 1st dense layer
        self.dense1 = nn.Sequential(
            nn.Linear(self.flatten_dim, self.hidden1), 
            nn.ReLU(True)
        )
        # the 2nd dense layer
        self.dense2 = nn.Sequential(
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(True)
        )
        # the output layer
        self.output = nn.Sequential(
            nn.Linear(self.hidden2, self.output_size),
        )
        self.softmax = nn.Softmax(dim=0)
    def forward(self,input_data):
        conv1 = self.conv1(input_data)
        conv2 = self.conv2(conv1)
        flatten_data = conv2.view(conv2.size(0),-1)
        dense1 = self.dense1(flatten_data)
        dense2 = self.dense2(dense1)
        output_data = self.output(dense2)
        output = self.softmax(output_data)

        return output

CNN = LeNet_5(output_size=10,flatten_dim=16*7*7,hidden_1=120,hidden_2=84) 
# CNN
optimizer = torch.optim.Adam(CNN.parameters(), lr=LEARNING_RATE)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
# training 
for epoch in range(1,TRAINING_STEPS+1):
    train_count = 0
    start_time= time.time()
    for (batch_data, batch_label) in train_loader:
        batch_data = batch_data
        batch_label = batch_label
        train_count +=1
        output = CNN(batch_data)
        loss = loss_func(output, batch_label)   # cross entropy loss
        optimizer.zero_grad()                   # clear gradients for this training step
        loss.backward()                         # backpropagation, compute gradients
        optimizer.step()                        # apply gradients
    for (test_data, test_label) in test_loader:
        test_data = test_data
        test_label = test_label
        test_output= CNN(test_data)
        pred_y = torch.max(test_output.cpu(), 1)[1].data.numpy()
        accuracy = float((pred_y == test_label.data.numpy()).astype(int).sum()) / float(test_label.size(0))
    end_time = time.time()
    duration = end_time-start_time
    print('Epoch: %04d' % epoch, '| train_steps: %02d' % train_count,'| test loss: %.4f' \
            % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy,'| duration: %.2fs'%duration)
    if epoch%5 == 0:
        # state = {'net':CNN.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(CNN,save_path+'/model.pth')
        # torch.onnx.export(CNN, batch_data, "Lenet5.onnx", verbose=True)
        print('The model has been saved after %04d'%epoch)