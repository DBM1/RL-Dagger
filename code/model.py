import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import math
import copy
import cv2
from sklearn.neighbors import KNeighborsClassifier
import random


class CNN(nn.Module):
    def __init__(self, conv1, conv2, conv3, liner, output):
        super(CNN, self).__init__()
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.liner = liner
        self.output = output

    def forward(self, data_batch):
        return self.output(self.liner(self.conv3(self.conv2(self.conv1(data_batch)))))


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


class BasicModel:
    def __init__(self):
        self.actions = [0, 1, 2, 3, 4, 5, 11, 12]
        pass

    def train(self, data_batch, label_batch):
        pass

    def predict(self, data_batch):
        pass


class CnnModel(BasicModel):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.d_model = 130
        self.d_output = 18
        self.cnn = CNN(ConvLayer(in_channel=1, out_channel=4, pool_size=2),
                       ConvLayer(in_channel=4, out_channel=8, pool_size=4),
                       ConvLayer(in_channel=8, out_channel=1, pool_size=2),
                       LinearLayer(self.d_model, dropout=0.1, n=3),
                       OutputLayer(self.d_model, self.d_output)).cuda()

    def train(self, data_batch, label_batch):
        data_batch = channel_fusion_images(data_batch)
        # label_batch = vectorization_integers(label_batch, size=self.d_output)
        epoch = 10
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
        print('CNN train finish')

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


class KnnModel(BasicModel):
    def __init__(self):
        super(KnnModel, self).__init__()
        self.knn = KNeighborsClassifier()

        self.is_first_step = True

    def train(self, data_batch, label_batch):
        if data_batch is None or len(data_batch) == 0:
            return
        data_batch = np.array(channel_fusion_images(data_batch))
        data_batch.squeeze(1)
        data_batch = data_batch.reshape(-1, 210 * 160)
        self.knn.fit(data_batch, label_batch)
        self.is_first_step = False
        print('KNN train finish')

    def predict(self, data_batch):
        result = []
        if self.is_first_step:
            for i in range(len(data_batch)):
                result.append(self.actions[random.randint(0, 7)])
        else:
            data_batch = np.array(channel_fusion_images(data_batch))
            data_batch.squeeze(1)
            data_batch = data_batch.reshape(-1, 210 * 160)
            result = self.knn.predict(data_batch)
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
