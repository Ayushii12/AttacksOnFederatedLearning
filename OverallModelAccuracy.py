##################################################################################################################################################################

#from os import chdir as cd
#cd('/content/drive/MyDrive/adversarial-robustness-toolbox-main/notebooks/')

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import torchvision

import os, sys
from os.path import abspath

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from art import config
from art.utils import load_dataset, get_file
from art.estimators.classification import PyTorchClassifier
from art.attacks.poisoning import FeatureCollisionAttack

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import copy

import time

import gc

np.random.seed(301)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
##################################################################################################################################################################

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')

print("Shape of x_train:",x_train.shape)
print("Shape of y_train:",y_train.shape)
print("Shape of x_test: ",x_test.shape)
print("Shape of y_test: ",y_test.shape)

x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

class_descr = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

##################################################################################################################################################################

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        # Download the model state_dict from the link: and run your code
        state_dict = torch.load(
            'resnet18.pt?dl=0', map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )
##################################################################################################################################################################

classifier_model = resnet18(pretrained=True)
classifier_model = classifier_model.to(device)
classifier_model.eval() 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_model.parameters(), lr=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

classifier = PyTorchClassifier(clip_values=(min_, max_), model=classifier_model, 
                               preprocessing=((0.4914, 0.4822, 0.4465),(0.2471, 0.2435, 0.2616)),nb_classes=10,input_shape=(3,32,32),loss=criterion,
                               optimizer=optimizer)

feature_layer = classifier.layer_names[-2]


##################################################################################################################################################################

no_of_clients = 50
clients = np.arange(no_of_clients)
images_per_client = 1000

# Holds images and labels of ALL clients
client_images=[]
client_labels=[]

for i in range(no_of_clients):
  start  = int(images_per_client*i)
  end    = int(images_per_client*(i+1))
  temp_x = x_train[start:end]
  temp_y = y_train[start:end]
  client_images.append(temp_x)
  client_labels.append(temp_y)
  print("client {} : {} - {}".format(i+1,start,end))

client_images = np.array(client_images)
client_labels = np.array(client_labels)

##################################################################################################################################################################

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

##################################################################################################################################################################

def test(model, weights):
  model.eval()

  with torch.no_grad():
    correct_pred = 0
    total_loss = 0.0
    for i in range(len(x_test)):
      inputs = x_test[i]
      labels = y_test[i]
      inputs =  torch.from_numpy(inputs).to(device)
      labels = torch.from_numpy(labels).to(device)
      inputs = inputs.reshape(1,3,32,32)
      labels = labels.reshape(1,10)

      output = model(inputs)

      loss = criterion(output, labels)
      total_loss += loss.item()

      _, pred = torch.max(output, dim = 1)
      _, actual_pred = torch.max(labels, dim=1)
      if pred == actual_pred:
        correct_pred += 1

      del inputs
      del labels
      gc.collect()
      torch.cuda.empty_cache()

  total_loss = total_loss/len(x_test)
  accuracy = 100*correct_pred / len(x_test)
  #print("Correct_pred: ", correct_pred)
  return accuracy, total_loss

##################################################################################################################################################################

valid_loss_min = np.Inf
for i in range(10):
  print("Iteration: ", i+1)

  idxs_users=np.random.choice(clients, size = (no_of_clients,), replace = False)
  global_weights = copy.deepcopy(classifier_model.state_dict())
  local_weights = []

  # Training each client
  for idx in idxs_users:
    classifier_model.load_state_dict(global_weights)
    if idx < 0:
        adv_train = np.vstack([client_images[idx], poison])
        adv_labels = np.vstack([client_labels[idx], poison_labels])
    else:
        adv_train = client_images[idx]
        adv_labels = client_labels[idx]
    
    classifier_model.train()
    classifier.fit(adv_train, adv_labels, nb_epochs=1, batch_size=128)
    local_weights.append(copy.deepcopy(classifier_model.state_dict()))
    #gc.collect()
    #torch.cuda.empty_cache()

  global_weights = FedAvg(local_weights)
  # Update global model 
  classifier_model.load_state_dict(global_weights)

  model_acc, val_loss = test(classifier_model, classifier_model.state_dict())
  print(model_acc)
  network_learned = val_loss < valid_loss_min
  
  if network_learned:
    valid_loss_min = val_loss
    torch.save(classifier_model.state_dict(), 'resnetAyu.pt')
    print('Improvement-Detected, save-model')

##################################################################################################################################################################

    
