import py7zr
from zipfile import ZipFile
from random import sample
import PIL.Image as Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from os import listdir
from os import path
import h5py
import wget

# data_path = IITD_Dataset.7z
filename = 'IITD_Dataset.7z'
# if (path.exists(filename)):
#     !rm $filename
#     print("File Removed!")
# print("Downloading Dataset...")
# wget.download(data_path, filename)
# print("Download Complete!")

with py7zr.SevenZipFile(filename, mode='r') as z:
    z.extractall()
    print("Extracted Dataset!")

# Processing the dataset
src_dir = 'ear/processed/221'
images_name = listdir(src_dir)
images_name_temp = []
subject_id = []
for img_ind in range(0, len(images_name)):
    if(not(images_name[img_ind]=='Thumbs.db')):
        images_name_temp.append(images_name[img_ind])
        subject_id.append(images_name[img_ind].split('_')[0])

images_name = images_name_temp
images_name_ordered = []
subject_id_ordered = []

sub_ind = sorted(range(len(subject_id)), key=lambda k: subject_id[k])   # Sorting the subject id
for pos, item in enumerate(sub_ind):
    images_name_ordered.append(images_name[item])
    subject_id_ordered.append(subject_id[item])

images_name = images_name_ordered
subject_id = subject_id_ordered

# print(subject_id)
# print(images_name)

img_ind = 0
ear_images = []
sub_labels = []
target_size = (180, 50)

for sub_ind in range (0,len(subject_id)):
    img_path = src_dir + '/' + images_name[sub_ind]
    ear_img = (plt.imread(img_path))

    ear_img = Image.open(img_path)
    ear_img = ear_img.resize(target_size, Image.ANTIALIAS)
    ear_img = np.array(ear_img).astype('float32')/255

    ear_images.append(ear_img)
    sub_labels.append(subject_id[sub_ind])
ear_images = np.array(ear_images)
sub_labels = np.array(sub_labels)

print("Shape of the dataset:", ear_images.shape)    
print("Shape of the labels:", sub_labels.shape)


X_train, X_test, y_train, y_test = train_test_split(ear_images, sub_labels, test_size=0.2, random_state=42)

print("Shape of the training dataset:", X_train.shape)  
print("Shape of the training labels:", y_train.shape)   
print("Shape of the testing dataset:", X_test.shape)    
print("Shape of the testing labels:", y_test.shape) 


import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional
import torch.optim as optim
from torchvision import models
from torchvision import transforms
# import torchsummary 
from torchinfo import summary

class Pytorch_BUS_Final_Model_C1(torch.nn.Module):
    def __init__(self, num_classes=221, num_filters=8, input_shape=(180,50,1)):
        super(Pytorch_BUS_Final_Model_C1, self).__init__()
        kernel_size = 3
        #Encoder Layer 1
        self.encoder_layer1_name = 'encoder_layer1'
        self.encoder_layer1_conv = torch.nn.Conv2d(in_channels=input_shape[2], out_channels=num_filters, kernel_size=kernel_size, padding='same')

        self.encoder_layer1_activation = torch.nn.ReLU()
        self.encoder_layer1_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        #Encoder Layer 2
        self.encoder_layer2_name = 'encoder_layer2'
        self.encoder_layer2_conv = torch.nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=kernel_size, padding='same')
        self.encoder_layer2_activation = torch.nn.ReLU()
        self.encoder_layer2_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder_layer2_batchnorm = torch.nn.BatchNorm2d(num_features=num_filters*2,eps=1e-3,momentum=0.99) 

        #Encoder Layer 3
        self.encoder_layer3_name = 'encoder_layer3'   
        self.encoder_layer3_conv = torch.nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4, kernel_size=kernel_size, padding='same')  
        self.encoder_layer3_activation = torch.nn.ReLU()
        self.encoder_layer3_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        #self.encoder_layer3_batchnorm = torch.nn.BatchNorm2d(num_features=num_filters*4,eps=1e-3,momentum=0.99)

        #Encoder Layer 4
        self.encoder_layer4_name = 'encoder_layer4'
        self.encoder_layer4_conv = torch.nn.Conv2d(in_channels=num_filters*4, out_channels=num_filters*8, kernel_size=kernel_size, padding='same')
        self.encoder_layer4_activation = torch.nn.ReLU()
        self.encoder_layer4_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder_layer4_batchnorm = torch.nn.BatchNorm2d(num_features=num_filters*8,eps=1e-3,momentum=0.99)

        #Encoder Layer 5
        self.encoder_layer5_name = 'encoder_layer5'
        self.encoder_layer5_conv = torch.nn.Conv2d(in_channels=num_filters*8, out_channels=num_filters*16, kernel_size=kernel_size, padding='same')
        self.encoder_layer5_activation = torch.nn.ReLU()
        self.encoder_layer5_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder_layer5_batchnorm = torch.nn.BatchNorm2d(num_features=num_filters*16,eps=1e-3,momentum=0.99)

        #Encoder Layer 6
        self.encoder_layer6_name = 'encoder_layer6'
        self.encoder_layer6_conv = torch.nn.Conv2d(in_channels=num_filters*16, out_channels=num_filters*32, kernel_size=kernel_size, padding='same')
        self.encoder_layer6_activation = torch.nn.ReLU()
        #self.encoder_layer6_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.encoder_layer6_batchnorm = torch.nn.BatchNorm2d(num_features=num_filters*32,eps=1e-3,momentum=0.99)
        
        #Dense Layers
        self.fc1_flatten = torch.nn.Flatten()
        self.fc1_linear = torch.nn.Linear(32*num_filters*(input_shape[0]//(2**5))*(input_shape[1]//(2**5)), out_features= num_classes)
        self.fc1_activation = torch.nn.Softmax()

    def forward(self,x):
        #Encoder Layer 1
        out = self.encoder_layer1_conv(x)
        out = self.encoder_layer1_activation(out)   
        out = self.encoder_layer1_pool(out)

        #Encoder Layer 2
        out = self.encoder_layer2_conv(out)
        out = self.encoder_layer2_activation(out)
        out = self.encoder_layer2_pool(out)
        out = self.encoder_layer2_batchnorm(out)

        #Encoder Layer 3
        out = self.encoder_layer3_conv(out)
        out = self.encoder_layer3_activation(out)
        out = self.encoder_layer3_pool(out)
        #out = self.encoder_layer3_batchnorm(out)

        #Encoder Layer 4
        out = self.encoder_layer4_conv(out)
        out = self.encoder_layer4_activation(out)
        out = self.encoder_layer4_pool(out)
        out = self.encoder_layer4_batchnorm(out)

        #Encoder Layer 5
        out = self.encoder_layer5_conv(out)
        out = self.encoder_layer5_activation(out)
        out = self.encoder_layer5_pool(out)
        out = self.encoder_layer5_batchnorm(out)

        #Encoder Layer 6
        out = self.encoder_layer6_conv(out)
        out = self.encoder_layer6_activation(out)
        #out = self.encoder_layer6_pool(out)
        out = self.encoder_layer6_batchnorm(out)

        #Dense Layers
        out = self.fc1_flatten(out)
        out = self.fc1_linear(out)
        out = self.fc1_activation(out)

        return out
    

pytorch_model = Pytorch_BUS_Final_Model_C1()
print(pytorch_model)
summary(pytorch_model, input_size=(1, 1, 180, 50))


