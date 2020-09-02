"""SimGNN class and runner."""

import glob
import torch
import random
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
import torchvision
import time
class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, number_of_labels):
    #def __init__(self):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(100352, 128)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)
        self.vggg = list(torchvision.models.vgg16(pretrained=True).cuda().children())[0][:24]
        self.backbone = torch.nn.Sequential(*self.vggg)
        #print('SB', self.backbone)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, im1,im2,im3,im4,im5,im6):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        #edge_index_1 = data["edge_index_1"]
        #edge_index_2 = data["edge_index_2"]
        #features_1 = data["features_1"]
        #features_2 = data["features_2"]
        #print(data.shape)
        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)
        im3 = im3.unsqueeze(0)
        im4 = im4.unsqueeze(0)
        im5 = im5.unsqueeze(0)
        im6 = im6.unsqueeze(0)
        f_g_1_1 = self.backbone(im1)
        f_g_1_2 = self.backbone(im2)
        f_g_1_3 = self.backbone(im3)
        f_g_2_1 = self.backbone(im4)
        f_g_2_2 = self.backbone(im5)
        f_g_2_3 = self.backbone(im6)
        edge_index_1 = torch.tensor([[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]],dtype=torch.long).cuda()
        #edge_index_1 = edge_index_1.t().contiguous()
        edge_index_2 = edge_index_1
        #print('zz11!', f_g_1_1.shape, f_g_2_3.shape, edge_index_1.shape)
        features_1 = torch.cat([f_g_1_1,f_g_1_2,f_g_1_3],dim=0)
        features_2 = torch.cat([f_g_2_1,f_g_2_2,f_g_2_3],dim=0)
        features_1 = features_1.view(3,-1)
        features_2 = features_1.view(3,-1)
        #features_1 = features_1.permute(2,3,1,0)
        #features_2 = features_2.permute(2,3,1,0)
        #print(features_1.shape, features_2.shape)
        #time.sleep(50)
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        #print('alalala')
        #time.sleep(5)
        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score

