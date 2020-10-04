import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import math
import copy
import cv2


class CNN(nn.Module):
    def __init__(self, conv1, conv2, liner, output):
        super(CNN, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.liner = liner
        self.output = output

    def forward(self, data_batch):
        return self.output(self.liner(self.conv2(self.conv1(data_batch))))


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, stride=1, padding=2, pool_size=None):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        if pool_size is not None:
            self.max_pool = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.max_pool is not None:
            x = self.max_pool(x)
        return x


class LinearLayer(nn.Module):
    def __init__(self, d_model, dropout, n=2):
        super(LinearLayer, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_model), n)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.dropout(x)
        for linear in self.linears:
            x = linear(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, d_model, d_output):
        super(OutputLayer, self).__init__()
        self.proj = nn.Linear(d_model, d_output)

    def forward(self, x):
        return func.softmax(self.proj(x), dim=-1)


def clones(module, n):
    # copy for n times
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Model:
    def __init__(self):
        self.d_model = 130
        self.d_output = 18
        self.cnn = CNN(ConvLayer(in_channel=1, out_channel=4, pool_size=8),
                       ConvLayer(in_channel=4, out_channel=1, pool_size=2),
                       LinearLayer(self.d_model, dropout=0.1, n=2),
                       OutputLayer(self.d_model, self.d_output)).cuda()

    def train(self, data_batch, label_batch):
        data_batch = channel_fusion_images(data_batch)
        # label_batch = vectorization_integers(label_batch, size=self.d_output)
        epoch = 1
        batch_size = 10
        nbatch = math.ceil(len(data_batch) / batch_size)
        lr = 0.01
        optimizer = torch.optim.Adam(self.cnn.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss().cuda()
        for _ in range(epoch):
            for i in range(nbatch):
                batch_x = torch.Tensor(data_batch[i * batch_size:(i + 1) * batch_size]).cuda()
                batch_y = torch.Tensor(label_batch[i * batch_size:(i + 1) * batch_size]).cuda()
                output = self.cnn(batch_x)
                loss = loss_func(output, batch_y.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(loss.cpu().detach().numpy())

    def predict(self, data_batch):
        result = []
        data_batch = channel_fusion_images(data_batch)
        batch_size = 10
        nbatch = math.ceil(len(data_batch) / batch_size)
        for i in range(nbatch):
            batch_x = torch.Tensor(data_batch[i * batch_size:(i + 1) * batch_size]).cuda()
            output = self.cnn(batch_x).cpu()
            max_index_sublist = output.argmax(1).numpy().tolist()
            result.extend(max_index_sublist)
        return result


def channel_fusion_images(image_list):
    image_gray_list = []
    for image in image_list:
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[np.newaxis, :]
        image_gray_list.append(img_gray)
    return image_gray_list


def vectorization_integers(integer_list, size=18):
    vector_list = []
    for integer in integer_list:
        vector = np.zeros(shape=size)
        vector[integer] = 1
        vector_list.append(vector)
    return vector_list
