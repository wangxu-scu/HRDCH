import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import numpy as np
class Embedding(torch.nn.Module):
        def __init__(self, data_class, binary_bits):
            super(Embedding, self).__init__()
            mid_num1 = 4096
            mid_num2 = 4096
            self.fc1 = nn.Linear(data_class, mid_num1)
            self.fc2 = nn.Linear(mid_num1, mid_num2)
            self.Embedding = nn.Linear(mid_num2, binary_bits)
            nn.init.uniform_( self.Embedding.weight, -1. / np.sqrt(np.float32(data_class)), 1. / np.sqrt(np.float32(data_class)) )

        def forward(self, x):
            out1 = F.relu(self.fc1(x))
            out2 = F.relu(self.fc2(out1))

            out3 = self.Embedding(out2).tanh()
            norm = torch.norm(out3, p=2, dim=1, keepdim=True)
            out3 = out3 / norm
            return  out3



class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=28*28, output_dim=20, tanh = True):
        super(ImgNN, self).__init__()
        mid_num1 = 4096
        mid_num2 = 4096
        self.tanh = tanh
        self.fc1 = nn.Linear(input_dim, mid_num1)
        self.fc2 = nn.Linear(mid_num1, mid_num2)

        self.fc3 = nn.Linear(mid_num2, output_dim, bias=False)
        nn.init.uniform_( self.fc3.weight, -1. / np.sqrt(np.float32(input_dim)), 1. / np.sqrt(np.float32(input_dim)) )


    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))

        out3 = self.fc3(out2).tanh() if self.tanh else self.fc3(out2)
        norm = torch.norm(out3, p=2, dim=1, keepdim=True)
        out3 = out3 / norm

        return  out3


class TextNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=28*28, output_dim=20, tanh = True):
        super(TextNN, self).__init__()
        mid_num1 = 4096
        mid_num2 = 4096
        self.tanh = tanh
        self.fc1 = nn.Linear(input_dim, mid_num1)
        self.fc2 = nn.Linear(mid_num1, mid_num2)

        self.fc3 = nn.Linear(mid_num2, output_dim, bias=False)
        nn.init.uniform_( self.fc3.weight, -1. / np.sqrt(np.float32(input_dim)), 1. / np.sqrt(np.float32(input_dim)) )


    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2).tanh() if self.tanh else self.fc3(out2)
        norm = torch.norm(out3, p=2, dim=1, keepdim=True)
        out3 = out3 / norm

        return  out3


class IDCM_NN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, img_input_dim=4096, text_input_dim=1024, output_dim =1024, num_class=10, tanh = True):
        super(IDCM_NN, self).__init__()
        self.img_net = ImgNN(img_input_dim, output_dim = output_dim, tanh = tanh)
        self.text_net = TextNN(text_input_dim, output_dim = output_dim, tanh = tanh)
        # W = torch.Tensor(output_dim, output_dim)
        # self.W = torch.nn.init.orthogonal_(W, gain=1)[:, 0: num_class]
        # self.W = self.W.clone().detach().cuda()
        # self.W.requires_grad=False
        self.W_1 = nn.Linear(output_dim, num_class, bias=False)
        self.W_2 = nn.Linear(output_dim, num_class, bias=False)

        # 用正交初始化
        nn.init.orthogonal_(self.W_1.weight, gain=1)
        nn.init.orthogonal_(self.W_2.weight, gain=1)

    def forward(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)

        view1_predict_logit = self.W_1(view1_feature)
        view2_predict_logit = self.W_2(view2_feature)

        return view1_feature, view2_feature,view1_predict_logit,view2_predict_logit
    def reset_parameters(self):
        for layer in self.img_net.children():
            layer.reset_parameters()
        for layer in self.text_net.children():
            layer.reset_parameters()





