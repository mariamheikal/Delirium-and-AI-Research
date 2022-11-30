
from __future__ import print_function, division
import math

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
from sklearn.utils.class_weight import compute_class_weight

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
from sklearn.datasets import make_classification
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_regression as MR
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

from typing import Callable

import pandas as pd
import torch
import torch.utils.data
import torchvision


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torch.utils.data.TensorDataset):
            return dataset.tensors[1]
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class MCC_Loss(nn.Module):
    """
    Calculates the proposed Matthews Correlation Coefficient-based loss.
    Args:
        inputs (torch.Tensor): 1-hot encoded predictions
        targets (torch.Tensor): 1-hot encoded ground truth
    """

    def __init__(self):
        super(MCC_Loss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        MCC = (TP.TN - FP.FN) / sqrt((TP+FP) . (TP+FN) . (TN+FP) . (TN+FN))
        where TP, TN, FP, and FN are elements in the confusion matrix.
        """
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )

        # Adding 1 to the denominator to avoid divide-by-zero errors.
        mcc = torch.div(numerator.sum(), denominator.sum() + 1.0)
        return 1 - mcc

def log_sum_exp(x):
    b, _ = torch.max(x, 1)
    # b.size() = [N, ], unsqueeze() required
    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
    # y.size() = [N, ], no need to squeeze()
    return y

def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)


def cross_entropy_with_weights(logits, target, weights=None):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = log_sum_exp(logits) - class_select(logits, target)
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss


class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':
            return cross_entropy_with_weights(input, target, weights).sum()
        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()
        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)
#weighted_bce=CrossEntropyLoss().cuda()
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

        f1 = 2*((1+0.75**2)*precision*recall) / ((0.75**2)*precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

f1_loss = F1_Loss().cuda()


class TransferFeatures(nn.Module):
    def __init__(self, original_model, classifier, model_name):
        super(TransferFeatures, self).__init__()

        self.features = original_model.parameters()
        self.classifier = classifier
        self.modelName = model_name

        # Freeze those weights
        for p in self.features:
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        # flatten network
        f = f.view(f.size(0), np.prod(f.shape[1:]))
        y = self.classifier(f)
        return y


data_dir = '/home/mas177/thesis_experiments/ImageDatasetSplit/'
data_dir_test = '/home/mas177/thesis_experiments/ImageDatasetTest/'


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

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, sampler=ImbalancedDatasetSampler(image_datasets[x]),
                                              num_workers=1)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data_transforms_test = {
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
image_datasets_test = {x: datasets.ImageFolder(os.path.join(data_dir_test, x),
                                          data_transforms_test[x])
                  for x in ['test']}
dataloaders_test = {x: torch.utils.data.DataLoader(image_datasets_test[x],batch_size=1,
                                             shuffle=True, num_workers=1)
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
                    #outputs = refit_model(inputs)
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

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


model_ft = models.googlenet(pretrained = True).to(device)

for params in list(model_ft.parameters()):
    params.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc=nn.Sequential(
    nn.Dropout(inplace=False),
    nn.Linear(in_features=num_ftrs, out_features=len(class_names), bias=True)
    )   
model_ft = model_ft.to(device)



criterion = f1_loss #nn.CrossEntropyLoss().cuda()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)
model_ft, ep_loss, ep_acc, res_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=100)

torch.save(model_ft,'TL_MODEL_MIMIC_F1_LOSS')

from sklearn.metrics import roc_auc_score


nb_classes = len(class_names)
probs=[]
pred=[]
confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['val']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        prob = F.softmax(outputs, dim=1)
        probs.append(prob)
        pred.append(preds)
        for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
final_preds=[]
for i in pred:
    for j in i:
        final_preds.append(int(j))
pred_prob=[]
k=0
for i in probs:
    for j in i:
        if final_preds[k]==1:
            pred_prob.append(float(j[1]))
        else:
            pred_prob.append(float(j[0]))
        k+=1

#print(probs)
#print(len(preds))


print('\n')
print('----------------------- VALIDATION DATA -----------------------------')
print('\n')

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
mcc=100*((TN*TP-FP*FN)/math.sqrt((TN+FN)*(FP+TP)*(TN+FP)*(FN+TP)))
print('Recall: ', recall)
print('Precision: ', precision)
print('Specificity: ',specificity)
print('NPV: ', npv)
print('MCC: ',mcc)




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
print('AUC: ',100*roc_auc_score(final_preds,pred_prob)) 
