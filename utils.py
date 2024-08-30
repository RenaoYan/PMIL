import numpy as np
import random
import torch
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def set_seed(num):
    torch.manual_seed(num)
    torch.cuda.manual_seed(num)
    np.random.seed(num)
    random.seed(num)
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    def __init__(self, model_path, patience=7, warmup_epoch=20, verbose=False, count_loss=True):
        self.patience = patience
        self.warmup_epoch = warmup_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = np.Inf
        self.model_path = model_path
        self.count_loss = count_loss

    def reset(self):
        self.counter = 0

    def __call__(self, epoch, val_loss, model, val_acc=None):
        flag = False
        if self.count_loss:
            if self.best_loss is None or val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(val_loss, model)
                self.counter = 0
                flag = True
        if val_acc is not None:
            if self.best_acc is None or val_acc >= self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint(val_acc, model, status='acc')
                self.counter = 0
                flag = True
        if flag:
            return self.counter
        self.counter += 1
        print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
        if self.counter >= self.patience and epoch >= self.warmup_epoch:
            self.early_stop = True
        return self.counter

    def save_checkpoint(self, score, model, status='loss'):
        """Saves model when validation loss or validation acc decrease."""
        if status == 'loss':
            pre_score = self.val_loss_min
            self.val_loss_min = score
        else:
            pre_score = self.val_acc_max
            self.val_acc_max = score
        torch.save(model.state_dict(), self.model_path)
        if self.verbose:
            print('Valid {} ({} --> {}).  Saving model ...{}'.format(status, pre_score, score, self.model_path))


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes, confusion_mat=False):
    targets = targets.numpy()
    _, pred = torch.max(logits, dim=1)
    pred = pred.numpy()
    acc = accuracy_score(targets, pred)
    f1 = f1_score(targets, pred, average='macro')

    probs = F.softmax(logits, dim=1)
    probs = probs.numpy()
    if len(np.unique(targets)) != num_classes:
        roc_auc = 0
    else:
        if num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true=targets, y_score=probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
        else:
            binary_labels = label_binarize(targets, classes=[i for i in range(num_classes)])
            valid_classes = np.where(np.any(binary_labels, axis=0))[0]
            binary_labels = binary_labels[:, valid_classes]
            valid_cls_probs = probs[:, valid_classes]
            fpr, tpr, _ = roc_curve(y_true=binary_labels.ravel(), y_score=valid_cls_probs.ravel())
            roc_auc = auc(fpr, tpr)
    if confusion_mat:
        mat = confusion_matrix(targets, pred)
        return acc, f1, roc_auc, mat
    return acc, f1, roc_auc


def plot_confusion_matrix(cmtx, num_classes, class_names=None, title='Confusion matrix', normalize=False,
                          cmap=plt.cm.Blues):
    if normalize:
        cmtx = cmtx.astype('float') / cmtx.sum(axis=1)[:, np.newaxis]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    figure = plt.figure()
    plt.imshow(cmtx, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    fmt = '.2f' if normalize else 'd'
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        plt.text(j, i, format(cmtx[i, j], fmt), horizontalalignment="center",
                 color="white" if cmtx[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return figure


def draw_metrics(ts_writer, name, num_class, loss, acc, auc, mat, f1, step):
    ts_writer.add_scalar("{}/loss".format(name), loss, step)
    ts_writer.add_scalar("{}/acc".format(name), acc, step)
    ts_writer.add_scalar("{}/auc".format(name), auc, step)
    ts_writer.add_scalar("{}/f1".format(name), f1, step)
    if mat is not None:
        ts_writer.add_figure("{}/confusion mat".format(name),
                             plot_confusion_matrix(cmtx=mat, num_classes=num_class), step)


def prepare_data(df, case_id, label_dict=None):
    df_case_id = df['case_id'].tolist()
    df_slide_id = df['slide_id'].tolist()
    df_label = df['label'].tolist()

    slide_id = []
    label = []
    for case_id_ in case_id:
        idx = df_case_id.index(case_id_)
        slide_id.append(df_slide_id[idx])
        label_ = df_label[idx]
        if label_dict is None:
            label.append(int(label_))
        else:
            label.append(label_dict[label_])
    return slide_id, label


def return_splits(csv_path, label_dict=None, label_csv=None, test=False):
    split_df = pd.read_csv(csv_path)
    train_id = split_df['train'].dropna().tolist()
    val_id = split_df['val'].dropna().tolist()
    if test:
        test_id = split_df['test'].dropna().tolist()
    if label_csv is None:
        train_label = split_df['train_label'].dropna().tolist()
        train_label = list(map(int, train_label))
        val_label = split_df['val_label'].dropna().tolist()
        val_label = list(map(int, val_label))
        if test:
            test_label = split_df['test_label'].dropna().tolist()
            test_label = list(map(int, test_label))
    else:
        df = pd.read_csv(label_csv)
        train_id, train_label = prepare_data(df, train_id, label_dict)
        val_id, val_label = prepare_data(df, val_id, label_dict)
        if test:
            test_id, test_label = prepare_data(df, test_id, label_dict)

    train_split = dict(zip(train_id, train_label))
    val_split = dict(zip(val_id, val_label))
    if test:
        test_split = dict(zip(test_id, test_label))
        return train_split, val_split, test_split
    return train_split, val_split


def shapley_value(search_indices, data, label, model, device, MIL_model='ABMIL', shuffle=True, shuffle_time=2,
                  subset_num=3):
    model.eval()
    with torch.no_grad():
        left_indices = [i for i in range(data.shape[1]) if i not in search_indices]
        random.shuffle(left_indices)
        left_data = data[:, left_indices, :]
        left_logits = []
        subset_data = [left_data[:,i::subset_num,:] for i in range(subset_num)]
        for _subset_data in subset_data:
            if 'ABMIL' in MIL_model:
                left_logit, _ = model(_subset_data.to(device))
            elif 'CLAM' in MIL_model:
                left_logit, _, results_dict = model(_subset_data.to(device), return_features=True)
            else:
                raise NotImplementedError
            left_logits.append(left_logit.cpu())
        cont = torch.zeros((data.shape[1], left_logit.shape[-1]))
        for i in search_indices:
            for j, _subset_data in enumerate(subset_data):
                x = torch.cat((data[:, i, :].unsqueeze(0), _subset_data), axis=1)
                for _ in range(shuffle_time):
                    if shuffle:
                        idx = torch.randperm(x.shape[1])
                        x = x[:, idx, :]
                    if 'ABMIL' in MIL_model:
                        logit, _ = model(x.to(device))
                    elif 'CLAM' in MIL_model:
                        logit, _, _ = model(x.to(device))
                    else:
                        raise NotImplementedError
                    cont[i] = cont[i] + logit.cpu() - left_logits[j]
        cont = cont / shuffle_time
        score = cont[search_indices, int(label)]
        score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
    return score

