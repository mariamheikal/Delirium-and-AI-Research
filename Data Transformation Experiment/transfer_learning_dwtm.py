from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns

import scipy
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from numpy import nan
from numpy import isnan
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from numpy import nan
from numpy import isnan
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression as LR, Lasso as LLR
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from random import shuffle
import time
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.datasets import fetch_openml


# Data augmentation and normalization for training
# Just normalization/content/content/HeartDiseasesdataset for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/home/mas177/thesis_experiments/ImageDatasetSplit_AUBMC/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data augmentation and normalization for training
# Just normalization/content/content/HeartDiseasesdataset for validation
data_transforms_test = {
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir_test = '/home/mas177/thesis_experiments/ImageDatasetTest_AUBMC/'
image_datasets_test = {x: datasets.ImageFolder(os.path.join(data_dir_test, x),
                                          data_transforms_test[x])
                  for x in ['test']}
dataloaders_test = {x: torch.utils.data.DataLoader(image_datasets_test[x], batch_size=16,
                                             shuffle=True, num_workers=4)
              for x in ['test']}
dataset_sizes_test = {x: len(image_datasets_test[x]) for x in ['test']}
class_names_test = image_datasets_test['test'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    res_dict = {}
    epoch_loss_list = []
    epoch_acc_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            epoch_loss_list.append(epoch_loss)
            epoch_acc_list.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            res_dict[phase] = epoch_acc_list
            


            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_loss_list, epoch_acc_list,res_dict

class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

f1_loss = F1_Loss().cuda()

#model_ft=models.googlenet(weights=torchvision.models.GoogLeNet_Weights.DEFAULT)

model_ft1 = models.googlenet(pretrained = True).to(device)
model_ft=torch.load('/home/mas177/thesis_experiments/TL_MODEL_MIMIC_F1_LOSS')
for params in list(model_ft.parameters()):
    params.requires_grad = False
num_ftrs = model_ft1.fc.in_features
model_ft.fc=nn.Sequential(
    nn.Dropout(inplace=False),
    nn.Linear(in_features=num_ftrs, out_features=len(class_names), bias=True)
    )   
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss().cuda()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft, ep_loss, ep_acc, res_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=200)
nb_classes = len(class_names)

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1


TP=confusion_matrix[1][1]
TN=confusion_matrix[0][0]
FP=confusion_matrix[0][1]
FN=confusion_matrix[1][0]
recall=100*TP/(TP+FN)
precision=100*TP/(TP+FP)
f1score=((recall*precision)/(recall+precision))*2
specificity=100*TN/(TN+FP)
accuracy=100*(TP+TN)/(TP+TN+FP+FN)
ppv=100*TP/(TP+FP)
npv= 100*TN/(FN+TN)
print('Accuracy: ', accuracy)
print('F1 Score: ', f1score)
print('Recall: ', recall)
print('Precision: ', precision)
print('Specificity: ',specificity)
print('NPV: ', npv)
                

print('\n')
print('\n')


from sklearn.metrics import roc_auc_score
import math 


nb_classes = len(class_names_test)
probs=[]
pred=[]
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders_test['test']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        prob = F.softmax(outputs, dim=1)
        probs.append(prob)
        pred.append(preds)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1



print('\n')
print('----------------------- TEST DATA -----------------------------')
print('\n')


final_preds=[]
for i in pred:
    for j in i:
        final_preds.append(int(j))
pred_prob=[]
k=0
for i in probs:
    for j in i:
        if k<len(final_preds) and final_preds[k]==1:
            pred_prob.append(float(j[1]))
        elif k<len(final_preds):
            pred_prob.append(float(j[0]))
        k+=1
print(len(final_preds))
print(len(pred_prob))
TP=confusion_matrix[1][1]
TN=confusion_matrix[0][0]
FP=confusion_matrix[0][1]
FN=confusion_matrix[1][0]
recall=100*TP/(TP+FN)
precision=100*TP/(TP+FP)
f1score=((recall*precision)/(recall+precision))*2
specificity=100*TN/(TN+FP)
accuracy=100*(TP+TN)/(TP+TN+FP+FN)
ppv=100*TP/(TP+FP)
npv= 100*TN/(FN+TN)
print(confusion_matrix)
print('Accuracy: ', accuracy)
print('F1 Score: ', f1score)
mcc=100*((TN*TP-FP*FN)/math.sqrt((TN+FN)*(FP+TP)*(TN+FP)*(FN+TP)))
print('Recall: ', recall)
print('Precision: ', precision)
print('Specificity: ',specificity)
print('NPV: ', npv)
print('MCC: ',mcc)
print('AUC: ',roc_auc_score(final_preds,pred_prob)) 