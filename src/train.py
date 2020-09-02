import numpy as np
import torch
import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
import torch.nn.functional as F
from PIL import Image
import torch.optim as optim
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

from torchvision.utils import save_image
from param_parser import parameter_parser
from model import SimGNN

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())


#yoyo = os.path.join(str_A, str_B, str_C)
#print(yoyo)

class GetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.landmarks)

    #fix below function.
    def __getitem__(self, idx):
        ext = '.png'
        g_1_center = int(self.landmarks.iloc[idx, 0])
        g_2_center = int(self.landmarks.iloc[idx, 1])
        img_name_A_2 = os.path.join(self.root_dir, 'fall_images_train/section1',str(self.landmarks.iloc[idx, 0])+ext)
        img_name_A_1 = os.path.join(self.root_dir, 'fall_images_train/section1',str(g_1_center - 1)+ ext)
        img_name_A_3 = os.path.join(self.root_dir, 'fall_images_train/section1',str(g_1_center + 1)+ ext)
        img_name_B_2 = os.path.join(self.root_dir, 'summer_images_train/section1',str(self.landmarks.iloc[idx, 1])+ext)
        img_name_B_1 = os.path.join(self.root_dir,'summer_images_train/section1' ,str(g_2_center - 1)+ ext)
        img_name_B_3 = os.path.join(self.root_dir,'summer_images_train/section1' ,str(g_2_center + 1)+ ext)
        label = self.landmarks.iloc[idx, 2] 
        #print(img_name_A, label)
        #hey = np.column_stack((self.transform(Image.open(img_name_A_1)), self.transform(Image.open(img_name_A_2))))
        return (self.transform(Image.open(img_name_A_1)),self.transform(Image.open(img_name_A_2)),self.transform(Image.open(img_name_A_3)),self.transform(Image.open(img_name_B_1)),self.transform(Image.open(img_name_B_2)),self.transform(Image.open(img_name_B_3)), label)


train_dataset = GetDataset(csv_file='/home/tourani/Desktop/code/simgnn_vpr/data/train.csv', root_dir='/home/tourani/Desktop/code/simgnn_vpr/data/', transform=transform)
test_dataset = GetDataset(csv_file='/home/tourani/Desktop/code/simgnn_vpr/data/test.csv', root_dir='/home/tourani/Desktop/code/simgnn_vpr/data/', transform=transform)
dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

def validate(net):
    
    net.eval()
    validation_dataloader = test_dataloader
    total = 0
    correct = 0
    for i, data in enumerate(validation_dataloader, 0):
        total = total + 1
        for z in range(1):
            im_1 = Variable(data[0][z]).cuda()
            im_2 = Variable(data[1][z]).cuda()
            im_3 = Variable(data[2][z]).cuda()
            im_4 = Variable(data[3][z]).cuda()
            im_5 = Variable(data[4][z]).cuda()
            im_6 = Variable(data[5][z]).cuda()
            cur_lab = Variable(data[6][z]).cuda()
            pred_label = net(im_1, im_2, im_3, im_4, im_5, im_6)
            pred_label = pred_label[0][0]
            #print(pred_label[0][0].shape, cur_lab.shape)
            if(pred_label > 0.5):
                pred_label = 1
            else:
                pred_label = 0
            correct = correct + (pred_label == cur_lab)
    print(correct, total)

def trainNet(net, batch_size, n_epochs, learning_rate):
    
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    train_loader = dataloader
    n_batches = len(train_loader)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,lr=0.001,
                                          weight_decay=5*10**-4)
    print('Beginning training\n')
    for epoch in range(n_epochs):
        net.train()
        running_loss = 0.0
        start_time = time.time()
        total_train_loss = 0
        cnat = 0
        for i, data in enumerate(train_loader, 0):
            #print(data[0].shape, data[1].shape,data[2].shape,data[3].shape,data[4].shape,data[5].shape,data[6].shape)
            #time.sleep(100)
            #print(len(data), data[0].shape, data[1].shape, data[2].shape, data[3].shape, data[4].shape, data[6].shape)
            #time.sleep(50)
            losses = 0
            optimizer.zero_grad()
            for z in range(4):
                im_1 = Variable(data[0][z]).cuda()
                im_2 = Variable(data[1][z]).cuda()
                im_3 = Variable(data[2][z]).cuda()
                im_4 = Variable(data[3][z]).cuda()
                im_5 = Variable(data[4][z]).cuda()
                im_6 = Variable(data[5][z]).cuda()
                cur_lab = Variable(data[6][z]).cuda()
                #optimizer.zero_grad()
                pred_label = net(im_1, im_2, im_3, im_4, im_5, im_6)
                losses = losses + torch.nn.functional.mse_loss(cur_lab, pred_label)
            cnat += 4
            if(cnat % 100 == 0):
                print(cnat)
            losses.backward()
            optimizer.step()
            running_loss = running_loss + losses
        print('Epoch: ', epoch, 'Loss: ', running_loss)
        if(epoch % 10 == 0):
            validate(net)
        
args = parameter_parser()
model_to_train = SimGNN(args,3).cuda()
for name, param in model_to_train.named_parameters():
    if param.requires_grad:
        if(name[0] == 'v' and name[1] == 'g'):
            param.requires_grad = False
            
for name, param in model_to_train.named_parameters():
    if param.requires_grad:
        print(name) 
#trainNet(model_to_train, batch_size=4, n_epochs=32000, learning_rate=0.0001)
validate(model_to_train)
