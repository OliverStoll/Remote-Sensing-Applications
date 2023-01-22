#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# In[108]:


import copy
import os
import numpy as np
import zipfile
from simple_downloader import download
from tqdm.notebook import tqdm
from pathlib import Path
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ## Define Customized Dataset in PyTorch

# In[109]:


class UCMerced(Dataset):
    def __init__(self, root_dir, img_transform=None, multilabel=False):

        self.root_dir = root_dir
        self.images_path = os.path.join(root_dir, "Images")
        self.class_names = sorted(
            [cl for cl in os.listdir(self.images_path) if not cl.startswith(".")]
        )
        self.img_paths, self.img_labels = self.init_dataset()
        self.img_transform = img_transform

        if multilabel:
            self.img_labels = self.read_multilabels()  # important for loss calculation
            self.img_labels = self.img_labels.astype(float)

    def init_dataset(self):
        img_paths, img_labels = [], []
        for cl_id, cl_name in enumerate(self.class_names):
            cl_path = os.path.join(self.images_path, cl_name)

            for img in sorted(os.listdir(cl_path)):
                img_path = os.path.join(cl_path, img)
                img_paths.append(img_path)
                img_labels.append(cl_id)

        return img_paths, img_labels

    def read_multilabels(self):  # TODO
        file_path = self.root_dir + "/multilabels/LandUse_Multilabeled.xlsx"

        df = pd.read_excel(file_path)
        df = df.drop("IMAGE\\LABEL", axis=1)
        labels_onehot = df.to_numpy()

        return labels_onehot

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            img = self.img_transform(img)

        return dict(img=img, label=label)

    def __len__(self):
        return len(self.img_paths)


# ## Define Customized PyTorch Model

# ## Helper Functions

# In[110]:


class MetricTracker(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[111]:


def get_device(cuda_int):
    """Get Cuda-Device. If cuda_int < 0 compute on CPU."""
    if cuda_int < 0:
        print("Computation on CPU")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(cuda_int))
        device = torch.device("cuda:{}".format(cuda_int))
    return device


# In[112]:


def get_dataset(root_dir, tr_transform, te_transform, set_sizes, seed=1, multilabel=False):
    """
    Parameter
    ---------
    root_dir     : path to UCMerced Dataset
    tr_transform : transformation for training data
    te_transform : transformation for training data
    set_sizes    : list of percentage of either train-test or train-val-test (sum to 100)

    Output
    ------
    sets for train and test, optionally also val if len(set_sizes)==3
    """
    ucm_dataset_tr = UCMerced(root_dir, img_transform=tr_transform, multilabel=multilabel)
    ucm_dataset_te = UCMerced(root_dir, img_transform=te_transform, multilabel=multilabel)
    idx_list = split_ucm_indices(set_sizes, seed=seed)

    train_set = Subset(ucm_dataset_tr, idx_list[0])
    test_set = Subset(ucm_dataset_te, idx_list[-1])

    if len(idx_list) > 2:
        val_set = Subset(ucm_dataset_te, idx_list[1])
        return train_set, val_set, test_set
    else:
        return train_set, test_set


# In[113]:


def split_ucm_indices(set_sizes, num_samples=2100, num_classes=21, seed=1):
    """Compute indices for a class-balanced train-(val)-test split for UCMerced."""
    cl_samples = int(num_samples / num_classes)
    assert sum(set_sizes) == 100
    split_indices = list(map(int, np.cumsum(set_sizes)[:-1] / 100 * cl_samples))
    # class_idx_mat d x N (row: classes, columns: idx of sample in dataset)
    dataset_idx = np.arange(0, num_samples)
    class_idx_mat = np.reshape(dataset_idx, (num_classes, cl_samples))
    # random shuffle class_wise idx (=> per row)
    np.random.seed(seed)
    np.apply_along_axis(np.random.shuffle, 1, class_idx_mat)
    # return indices for splits (2 or 3)
    idx_list = np.hsplit(class_idx_mat, split_indices)
    # flatten set idx
    return list(map(lambda x: x.flatten(), idx_list))


# In[114]:


def pretty_classification_report_print(report, class_names):
    N = len(class_names)
    df = pd.DataFrame(report).round(decimals=2)
    df = df.rename(columns=dict(zip(list(map(str, range(N))), testset.dataset.class_names))).T
    df[["support"]] = df[["support"]].astype(int)
    return df


# In[115]:


def prettify_confusion_matrix(conf_mat, class_names):
    plt.subplots(1, 1, figsize=(11, 7))
    sns.heatmap(
        conf_mat,
        cmap="viridis",
        fmt="g",
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
    )


# ## Training and Evaluation Functions

# In[116]:


def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, early_stop=False):
    train_losses, val_losses = [], []
    accuracy_scores = []
    best_model = copy.deepcopy(model)
    best_acc = 0
    best_epoch = 1

    for epoch in range(1, epochs + 1):

        print("Epoch {}/{}".format(epoch, epochs))
        print("-" * 10)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, report, _ = val_epoch(model, val_loader, criterion, device)
        overall_acc = report["accuracy"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracy_scores.append(overall_acc)

        if best_acc < overall_acc:
            best_acc = overall_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if epoch - best_epoch > 10 and early_stop:
            break

    return best_model, train_losses, val_losses, accuracy_scores


# In[117]:


def train_epoch(model, train_loader, optimizer, criterion, device):
    loss_tracker = MetricTracker()
    acc_tracker = MetricTracker()
    model.train()

    tqdm_bar = tqdm(train_loader, desc="Training: ")
    for batch in tqdm_bar:

        images = batch["img"].to(device)
        labels = batch["label"].to(device)
        batch_size = images.size(0)
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probs = F.softmax(logits, dim=1)
        loss_tracker.update(loss.item(), batch_size)

        _, predicted = torch.max(probs.data, 1)
        batch_acc = (predicted == labels).sum().item() / batch_size
        acc_tracker.update(batch_acc, batch_size)
        tqdm_bar.set_postfix(loss=loss_tracker.avg, accuracy=acc_tracker.avg)

    return loss_tracker.avg


# In[118]:


def val_epoch(model, val_loader, criterion, device):
    loss_tracker = MetricTracker()
    acc_tracker = MetricTracker()
    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        tqdm_bar = tqdm(val_loader, desc="Validation: ")
        for batch in tqdm_bar:

            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            batch_size = images.size(0)

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            loss = criterion(logits, labels)
            loss_tracker.update(loss.item(), batch_size)

            _, predicted = torch.max(probs.data, 1)
            batch_acc = (predicted == labels).sum().item() / batch_size
            acc_tracker.update(batch_acc, batch_size)

            y_pred += predicted.tolist()
            y_true += labels.tolist()
            tqdm_bar.set_postfix(loss=loss_tracker.avg, accuracy=acc_tracker.avg)

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    conf_mat = confusion_matrix(y_true, y_pred, normalize="true")

    return loss_tracker.avg, report, conf_mat


# # Training on UCMerced Dataset

# ## Download UCMerced Dataset from TUB-Cloud
# 
# Following workflow from Lab01, creating directory "./data", downloading UCMerced dataset zip-file and unzipping it.

# In[119]:


download_dir = Path("./data")
download_dir.mkdir(exist_ok=True)


# In[120]:


TUB_URL = "https://tubcloud.tu-berlin.de/s/H4QHX5GPDY6wDog/download/UCMerced_LandUse.zip"
output_file = download(TUB_URL, "./data/")


# In[121]:


zipf = zipfile.ZipFile(output_file)
zipf.extractall(path="data")


# ## Main Hyperparamter

# In[122]:


cuda_device = get_device(0)


# In[123]:


batch_size = 64
learning_rate = 0.001
epochs = 20
num_cls = 21


# ## Train- and Testset Transformation (i.e., Data Augmentation)

# In[124]:


ucm_mean = [0.595425, 0.3518577, 0.3225522]
ucm_std = [0.19303136, 0.12492529, 0.10577361]

tr_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ucm_mean, std=ucm_std),
    ]
)

te_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ucm_mean, std=ucm_std),
    ]
)


# ## Initializing Train-, Val-, Testset and Dataloader

# In[125]:


trainset, valset, testset = get_dataset(
    "./data/UCMerced_LandUse",
    tr_transform=tr_transform,
    te_transform=te_transform,
    set_sizes=[70, 10, 20],
    multilabel=True,
)


# In[126]:


train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)


# ## Initialize Model, Loss-Function and Optimizer

# When using models that are pretrained on a different, usually bigger dataset (e.g. on the popular Computer Vision dataset ImageNet [1]) for a so-called downstream task in which the pretrained model is fine-tuned on the target dataset, we speak about Transfer Learning.
# 
# In this practice, try to use a pretrained version of the predefined `resnet18` model from Pytorch `models` library. You can achieve this by providing pretrained model weights to the weights parameter at initialization of the model. Use the following weights from the `torchvision` library: `torchvision.models.ResNet18_Weights.DEFAULT`. Finetune it on the train set of UCMerced dataset and evaluate the new model.
# 
# **Hint**: ImageNet consists of 1000 classes, therefore the last layer of the pretrained model needs to be alternated.
# 
# for more information check the pytorch documentation:
# 
# https://pytorch.org/docs/stable/torchvision/models.html
# 
# [1] https://www.image-net.org/index.php

# In[127]:


# Add necessary functionality for multi label!!!!

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones([17])).to(cuda_device)
eval_accuracies = []


# In[128]:


resnet_pretrained = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet_pretrained.fc = nn.Linear(512, num_cls)
resnet_pretrained.to(cuda_device)
optimizer = optim.SGD(
    resnet_pretrained.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=0.0001,
    nesterov=True,
)


# In[129]:


best_model, train_losses, val_losses, accuracy_scores = train(
    resnet_pretrained,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs=epochs,
    device=cuda_device,
)
eval_accuracies.append(accuracy_scores)

