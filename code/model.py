# import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import copy
import cv2


class CNN(nn.Module):
    def __init__(self, conv, liner, output):
        super(CNN, self).__init__()
        self.conv = conv
        self.liner = liner
        self.output = output

    def forward(self, data_batch):
        return self.output(self.liner(self.conv(data_batch)))


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()


class LinearLayer(nn.Module):
    def __init__(self, d_model, dropout, n=2):
        super(LinearLayer, self).__init__()
        self.linears = clones(nn.Linear(d_model, d_model), n)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        for linear in self.linears:
            x = linear(x)
        return x


class OutputLayer(nn.Module):
    def __init__(self, d_model, d_output):
        super(OutputLayer, self).__init__()
        self.proj = nn.Linear(d_model, d_output)

    def forward(self, x):
        return func.log_softmax(self.proj(x), dim=-1)


def clones(module, n):
    # copy for n times
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Model:
    def __init__(self):
        self.d_model = 128
        self.d_output = 18
        self.cnn = CNN(ConvLayer(),
                       LinearLayer(self.d_model, dropout=0., n=2),
                       OutputLayer(self.d_model, self.d_output))

    def train(self, data_batch, label_batch):
        data_batch = channel_fusion_images(data_batch)
        label_batch = vectorization_integers(label_batch, size=self.d_output)
        pass

    def predict(self, data_batch):
        data_batch = channel_fusion_images(data_batch)
        pass


def channel_fusion_images(image_list):
    image_gray_list = []
    for image in image_list:
        img_gray = np.ndarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        image_gray_list.append(img_gray)
    return image_gray_list


def vectorization_integers(integer_list, size=18):
    vector_list = []
    for integer in integer_list:
        vector = np.zeros(shape=size)
        vector[integer] = 1
        vector_list.append(vector)
    return vector_list
