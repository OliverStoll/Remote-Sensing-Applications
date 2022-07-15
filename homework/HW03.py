import copy
import json
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
from sklearn.metrics import f1_score, average_precision_score, ConfusionMatrixDisplay, confusion_matrix

# ## Define Customized Dataset in PyTorch
import warnings

warnings.filterwarnings("ignore")
global_images = []
global_epoch = 0
global_batch = 0

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

    def read_multilabels(self):
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


def get_device(cuda_int):
    """Get Cuda-Device. If cuda_int < 0 compute on CPU."""
    if cuda_int < 0:
        print("Computation on CPU")
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(cuda_int))
        device = torch.device("cuda:{}".format(cuda_int))
    return device


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


def pretty_classification_report_print(report, class_names):
    N = len(class_names)
    df = pd.DataFrame(report).round(decimals=2)
    df = df.rename(columns=dict(zip(list(map(str, range(N))), testset.dataset.class_names))).T
    df[["support"]] = df[["support"]].astype(int)
    return df


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

def get_eval_metrics(labels_eval, predicted):
    metrics = {
        "micro_f1": f1_score(labels_eval, predicted, average="micro", zero_division=0),
        "macro_f1": f1_score(labels_eval, predicted, average="macro", zero_division=0),
        "micro_map": average_precision_score(labels_eval, predicted, average="micro"),
        "macro_map": average_precision_score(labels_eval, predicted, average="macro")
    }
    return metrics


def train(model, train_loader, val_loader, optimizer, criterion, epochs, device, early_stop=False):
    all_metrics = {"train": {"micro_f1": [], "macro_f1": [], "micro_map": [], "macro_map": []},
                   "val": {"micro_f1": [], "macro_f1": [], "micro_map": [], "macro_map": []}
                   }

    for epoch in range(1, epochs + 1):
        print("Epoch {}/{}".format(epoch, epochs))
        print("-" * 10)

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_metrics = val_epoch(model, val_loader, criterion, device)

        for metric in train_metrics.keys():
            all_metrics["train"][metric] += [train_metrics[metric]]
        for metric in val_metrics.keys():
            all_metrics["val"][metric] += [val_metrics[metric]]

    return all_metrics, model


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    loss_tracker = MetricTracker()
    model.train()

    metrics_total = {"micro_f1": [], "macro_f1": [], "micro_map": [], "macro_map": []}

    tqdm_bar = tqdm(train_loader, desc="Training: ")
    batch_num = 1
    for batch in tqdm_bar:
        images = batch["img"].to(device)
        if epoch == 1 and batch_num == 1:
            # save first batch of images for visualization as .png
            for i, image in enumerate(images):
                # save image as .png
                image_path = f"./output/images/{i}"
                image = image.cpu().numpy().transpose(1, 2, 0)
                # rescale image to [0, 1]
                image = (image - image.min()) / (image.max() - image.min())
                image = image * 255
                image = image.astype(np.uint8)
                image = Image.fromarray(image)
                image.save(image_path + ".png")

        labels_eval = batch["label"].numpy()
        labels = batch["label"].to(device)
        batch_size = images.size(0)
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(logits)

        loss_tracker.update(loss.item(), batch_size)

        # determine predictions by thresholding at 0.5
        predicted = (probs.cpu().detach().numpy() > 0.5).astype(int)

        # calculate scores
        metrics = get_eval_metrics(labels_eval, predicted)

        metrics_total["micro_f1"].append(metrics["micro_f1"])
        metrics_total["macro_f1"].append(metrics["macro_f1"])
        metrics_total["micro_map"].append(metrics["micro_map"])
        metrics_total["macro_map"].append(metrics["macro_map"])

    final_metrics = {
        "micro_f1": np.mean(metrics_total["micro_f1"]),
        "macro_f1": np.mean(metrics_total["macro_f1"]),
        "micro_map": np.mean(metrics_total["micro_map"]),
        "macro_map": np.mean(metrics_total["macro_map"])
    }
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}", end=" - ")

    return final_metrics


def val_epoch(model, val_loader, criterion, device):
    loss_tracker = MetricTracker()
    acc_tracker = MetricTracker()
    model.eval()

    y_pred = []
    y_true = []
    metrics_total = {"micro_f1": [], "macro_f1": [], "micro_map": [], "macro_map": []}

    with torch.no_grad():
        tqdm_bar = tqdm(val_loader, desc="Validation: ")
        for batch in tqdm_bar:
            images = batch["img"].to(device)
            labels_eval = batch["label"].numpy()
            labels = batch["label"].to(device)
            batch_size = images.size(0)

            logits = model(images)
            probs = torch.sigmoid(logits)
            loss = criterion(logits, labels)

            # determine predictions by thresholding at 0.5
            predicted = (probs.cpu().detach().numpy() > 0.5).astype(int)

            # calculate scores
            metrics = get_eval_metrics(labels_eval, predicted)

            metrics_total["micro_f1"].append(metrics["micro_f1"])
            metrics_total["macro_f1"].append(metrics["macro_f1"])
            metrics_total["micro_map"].append(metrics["micro_map"])
            metrics_total["macro_map"].append(metrics["macro_map"])

            tqdm_bar.set_postfix(loss=loss_tracker.avg, accuracy=acc_tracker.avg)

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    # plot confusion matrix
    cm_display = ConfusionMatrixDisplay.from_estimator(model, y_true, y_pred)
    cm_display.plot(cmap=plt.cm.Blues)

    final_metrics = {
        "micro_f1": np.mean(metrics_total["micro_f1"]),
        "macro_f1": np.mean(metrics_total["macro_f1"]),
        "micro_map": np.mean(metrics_total["micro_map"]),
        "macro_map": np.mean(metrics_total["macro_map"])
    }
    for key, value in final_metrics.items():
        print(f"{key}: {value:.4f}", end=" - ")

    return final_metrics


# # Training on UCMerced Dataset

# ## Download UCMerced Dataset from TUB-Cloud
#
# Following workflow from Lab01, creating directory "./data", downloading UCMerced dataset zip-file and unzipping it.


download_dir = Path("./data")
download_dir.mkdir(exist_ok=True)

TUB_URL = "https://tubcloud.tu-berlin.de/s/H4QHX5GPDY6wDog/download/UCMerced_LandUse.zip"
output_file = download(TUB_URL, "./data/")

zipf = zipfile.ZipFile(output_file)
zipf.extractall(path="data")

# ## Main Hyperparamter


cuda_device = get_device(0)
num_cls = 17
ucm_mean = [0.595425, 0.3518577, 0.3225522]
ucm_std = [0.19303136, 0.12492529, 0.10577361]

batch_size = 64
learning_rate = 0.001
epochs = 10


# ## Train- and Testset Transformation (i.e., Data Augmentation)

def eval_all(tr_transform):
    te_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ucm_mean, std=ucm_std),
        ]
    )

    trainset, valset, testset = get_dataset(
        "./data/UCMerced_LandUse",
        tr_transform=tr_transform,
        te_transform=te_transform,
        set_sizes=[70, 10, 20],
        multilabel=True,
    )

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0] * 17)).to(cuda_device)

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

    metrics, model = train(
        resnet_pretrained,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        epochs=epochs,
        device=cuda_device,
    )

    # plot confusion matrix
    # cm_display = ConfusionMatrixDisplay.from_estimator(model,)
    # cm_display.plot(cmap=plt.cm.Blues)

    return metrics


no_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ucm_mean, std=ucm_std),
    ]
)

basic_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ucm_mean, std=ucm_std),
        transforms.RandomAffine(10),
        transforms.RandomAutocontrast()
    ]
)

my_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ucm_mean, std=ucm_std),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30)
    ]
)

rand_transform = transforms.Compose(
[
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ucm_mean, std=ucm_std),
        transforms.RandAugment()
    ]
)

transforms_dict = {
    # "no_transform": no_transform,
    # "basic_transform": basic_transform,
    # "my_transform": my_transform,
    "rand_transform": rand_transform
}

for name, transform in transforms_dict.items():
    print(f"Evaluating {name}")
    metrics = eval_all(transform)
    print(f"{name}: {metrics}")
    # dump metrics to json file
    with open(f"./output/{name}_metrics.json", "w") as f:
        json.dump(metrics, f)


# ## Visualize Results
