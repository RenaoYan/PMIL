import os
import glob
import argparse
import timm
import torch.nn as nn
import torch.optim
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.MIL_models import ABMIL, CLAM_MB, CLAM_SB, MeanMIL, MaxMIL, DSMIL
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description='MIL main parameters')

    # General params.
    parser.add_argument('--experiment_name', type=str, default='PMIL_fold0', help='experiment name')
    parser.add_argument('--MIL_model', type=str, default='ABMIL',
                        choices=['ABMIL', 'CLAM_SB', 'CLAM_MB', 'MeanMIL', 'MaxMIL', 'DSMIL'],
                        help='MIL model to use')
    parser.add_argument('--metric2save', type=str, default='f1_auc',
                        choices=['acc', 'f1', 'auc', 'acc_auc', 'f1_auc', 'loss'],
                        help='metrics to save best model')
    parser.add_argument('--device_ids', type=str, default=0, help='gpu devices for training')
    parser.add_argument('--seed', type=int, default=3721, help='random seed')
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--num_classes', type=int, default=2, help='classification number')

    # Progressive pseudo bag augmentation params.
    parser.add_argument('--split_data', action='store_false', help='use data split')
    parser.add_argument('--search_rate', type=int, default=10, help='search rate')
    parser.add_argument('--sample_rate', type=int, default=3, help='subset sample rate')
    parser.add_argument('--max_pseudo_num', type=int, default=10, help='max pseudo bag number')
    parser.add_argument('--pseudo_step', type=int, default=2, help='pseudo bag number step')
    parser.add_argument('--metrics', type=str, default='shap', choices=['random', 'attn', 'shap'],
                        help='IIS estimation metrics to sort')

    # MIL training params.
    parser.add_argument('--rounds', type=int, default=10, help='rounds to train')
    parser.add_argument('--epochs', type=int, default=200, help='MIL epochs to train in each round')
    parser.add_argument('--patience', type=int, default=20, help='MIL epochs to early stop')
    parser.add_argument('--lr_patience', type=int, default=7, help='MIL epochs to adjust lr')
    parser.add_argument('--max_lr', type=float, default=1e-3, help='MIL max learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='MIL min learning rate')

    # dir params.
    parser.add_argument('--csv_dir', type=str,
                        default='./csv/camelyon16',
                        help='csv dir to split data')
    parser.add_argument('--feat_dir', type=str,
                        default='/DATA/feat',
                        help='train/val/test dir for features')
    parser.add_argument('--ckpt_dir', type=str,
                        default='./ckpt/camelyon16',
                        help='dir to save models')
    parser.add_argument('--test', action='store_false', help='use test dataset')
    parser.add_argument('--logger_dir', type=str,
                        default='./logger/camelyon16',
                        help='tensorboard dir')
    args = parser.parse_args()
    return args


class MILDataset(Dataset):
    def __init__(self, split, feat_dir):
        self.slide_ids = list(split.keys())
        self.labels = list(split.values())
        self.feat_dir = feat_dir
        self.feat_files = self.get_feat()

    def get_labels(self):
        return self.labels

    def get_feat(self):
        feat_files = {}
        for slide_id in self.slide_ids:
            feat_paths = glob.glob(os.path.join(self.feat_dir, slide_id + '.pt*'))
            slide_feats = []
            for feat_path in feat_paths:
                slide_feats.append(feat_path)
            feat_files[slide_id] = slide_feats
        return feat_files

    def __getitem__(self, idx):
        slide_name = self.slide_ids[idx]
        target = self.labels[idx]
        feat_files = self.feat_files[slide_name]
        feats = torch.Tensor()
        for feat_file in feat_files:
            feat = torch.load(feat_file, map_location='cpu')
            try:
                feat = torch.from_numpy(feat)
            except:
                pass
            feats = torch.cat((feats, feat), dim=0)

        sample = {'slide_id': slide_name, 'feat': feats, 'target': target}
        return sample

    def __len__(self):
        return len(self.slide_ids)


def MIL_train_epoch(round_id, epoch, model, optimizer, loader, criterion, device, num_classes, model_suffix='ABMIL',
                    split=None):
    model.train()
    loss_all = 0.
    logits = torch.Tensor()
    targets = torch.Tensor()
    with tqdm(total=len(loader)) as pbar:
        for _, sample in enumerate(loader):
            optimizer.zero_grad()
            slide_id, feat, target = sample['slide_id'], sample['feat'], sample['target']
            if len(feat[0]) == 0:
                pbar.update(1)
                continue
            feat = feat.to(device)
            target = target.to(device)
            if split is None:
                if 'ABMIL' in model_suffix:
                    logit, _ = model(feat)
                    loss = criterion(logit, target.long())
                elif 'DSMIL' in model_suffix:
                    bag_weight = 0.5
                    patch_pred, logit, _, _ = model(feat)
                    patch_pred, _ = torch.max(patch_pred, 0)
                    loss = bag_weight * criterion(patch_pred.view(1, -1), target.long()) \
                           + (1 - bag_weight) * criterion(logit, target.long())
                elif 'CLAM' in model_suffix:
                    bag_weight = 0.7
                    logit, _, instance_dict = model(feat, target, instance_eval=True)
                    instance_loss = instance_dict['instance_loss']
                    loss = bag_weight * criterion(logit, target.long()) + (1 - bag_weight) * instance_loss
                else:
                    logit = model(feat)
                    loss = criterion(logit, target.long())

                # calculate metrics
                logits = torch.cat((logits, logit.detach().cpu()), dim=0)
                targets = torch.cat((targets, target.cpu()), dim=0)
                loss_all += loss.detach().item() * len(target)
                # loss backward
                loss.backward()
                optimizer.step()
            else:
                split_index = split[slide_id[0]]
                for i in range(len(split_index)):
                    idx = split_index[i]
                    # np.random.shuffle(idx)
                    bag_feat = feat[:, idx, :]
                    if 'ABMIL' in model_suffix:
                        bag_logit, _ = model(bag_feat)
                        bag_loss = criterion(bag_logit, target.long())
                    elif 'DSMIL' in model_suffix:
                        bag_weight = 0.5
                        bag_patch_pred, bag_logit, _, _ = model(feat)
                        bag_patch_pred, _ = torch.max(bag_patch_pred, 0)
                        bag_loss = bag_weight * criterion(bag_patch_pred.view(1, -1), target.long()) \
                                   + (1 - bag_weight) * criterion(bag_logit, target.long())
                    elif 'CLAM' in model_suffix:
                        bag_weight = 0.7
                        bag_logit, _, instance_dict = model(bag_feat, target, instance_eval=True)
                        instance_loss = instance_dict['instance_loss']
                        bag_loss = bag_weight * criterion(bag_logit, target.long()) + (1 - bag_weight) * instance_loss
                    else:
                        logit = model(feat)
                        bag_loss = criterion(logit, target.long())
                    logits = torch.cat((logits, bag_logit.detach().cpu()), dim=0)
                    targets = torch.cat((targets, target.cpu()), dim=0)
                    loss_all += bag_loss.detach().item() * len(target) / (len(split_index))
                    # loss backward
                    bag_loss.backward()
                    optimizer.step()
            acc, f1, roc_auc = calculate_metrics(logits, targets, num_classes)

            lr = optimizer.param_groups[0]['lr']

            if split is None:
                pbar.set_description(
                    '[Round:{}, Epoch:{}] lr:{:.5f}, loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                        .format(round_id, epoch, lr, loss_all / len(targets), acc, roc_auc, f1))
            else:
                pbar.set_description(
                    '[Round:{}, Epoch:{}] split:{}, lr:{:.5f}, loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                        .format(round_id, epoch, len(split_index), lr, loss_all / len(targets), acc, roc_auc, f1))
            pbar.update(1)
    acc, f1, roc_auc, mat = calculate_metrics(logits, targets, num_classes, confusion_mat=True)
    # print(mat)
    return loss_all / len(targets), acc, roc_auc, f1, mat


def MIL_pred(round_id, model, loader, criterion, device, num_classes, model_suffix='ABMIL', status='Val'):
    model.eval()
    loss_all = 0.
    logits = torch.Tensor()
    targets = torch.Tensor()
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for _, sample in enumerate(loader):
                slide_id, feat, target = sample['slide_id'], sample['feat'], sample['target']
                if len(feat[0]) == 0:
                    pbar.update(1)
                    continue
                feat = feat.to(device)
                target = target.to(device)
                if 'ABMIL' in model_suffix:
                    logit, _ = model(feat)
                elif 'DSMIL' in model_suffix:
                    _, logit, _, _ = model(feat)
                elif 'CLAM' in model_suffix:
                    logit, _, _ = model(feat, target)
                else:
                    logit = model(feat)

                # calculate metrics
                loss = criterion(logit, target.long())
                logits = torch.cat((logits, logit.detach().cpu()), dim=0)
                targets = torch.cat((targets, target.cpu()), dim=0)
                loss_all += loss.item() * len(target)
                acc, f1, roc_auc = calculate_metrics(logits, targets, num_classes)

                pbar.set_description('[{} Round:{}] loss:{:.4f}, acc:{:.4f}, auc:{:.4f}, f1:{:.4f}'
                                     .format(status, round_id, loss_all / len(targets), acc, roc_auc, f1))
                pbar.update(1)
    acc, f1, roc_auc, mat = calculate_metrics(logits, targets, num_classes, confusion_mat=True)
    print(mat)
    return loss_all / len(targets), acc, roc_auc, f1, mat


def split_slide_data(split_num, model, model_path, loader, device, model_suffix='ABMIL', prev_split=None,
                     metrics='attn', search_rate=10, sample_rate=3):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        metrics='random'
    model.eval()
    split = {}
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for _, sample in enumerate(loader):
                slide_id, feat, target = sample['slide_id'], sample['feat'], sample['target']
                if len(feat[0]) == 0:
                    pbar.update(1)
                    continue
                feat = feat.to(device)
                target = target.to(device)
                if metrics == 'random':
                    attn_index = np.arange(feat.shape[1])
                    np.random.shuffle(attn_index)
                else:
                    if prev_split is None:
                        if 'ABMIL' in model_suffix:
                            _, attn = model(feat)
                        elif 'CLAM' in model_suffix:
                            _, attn, _ = model(feat, target)
                        else:
                            attn = np.zeros((1, feat.shape[1]))
                    else:
                        attn = np.zeros((1, feat.shape[1]))
                        split_index = prev_split[slide_id[0]]
                        for i in range(len(split_index)):
                            idx = split_index[i]
                            np.random.shuffle(idx)
                            bag_feat = feat[:, idx, :]
                            if 'ABMIL' in model_suffix:
                                _, bag_attn = model(bag_feat)
                            elif 'CLAM' in model_suffix:
                                _, bag_attn, _ = model(bag_feat, target, instance_eval=True)
                            else:
                                bag_attn = np.zeros((1, feat.shape[1]))
                            attn[:, idx] = bag_attn[0]
                    score = attn[0]
                    attn_index = np.argsort(-score)
                    search_num = int(min(split_num * search_rate, len(score) / 2))
                    if metrics == 'shap':
                        search_indices = attn_index[:search_num]
                        shap = shapley_value(search_indices, feat, target, model, device, model_suffix, subset_num=sample_rate)
                        ptopk_indices = search_indices[np.argsort(-shap)]
                        left_indices = attn_index[search_num:]
                        attn_index = ptopk_indices.tolist() + left_indices.tolist()
                    elif metrics == 'cont':
                        search_indices = attn_index[:search_num]
                        cont = contribution(search_indices, feat, target, model, device, model_suffix)
                        ptopk_indices = search_indices[np.argsort(-cont)]
                        left_indices = attn_index[search_num:]
                        attn_index = ptopk_indices.tolist() + left_indices.tolist()
                split[slide_id[0]] = [attn_index[i::split_num] for i in range(split_num)]
                pbar.update(1)
    return split


if __name__ == '__main__':
    args = get_args()

    # set device
    device = torch.device('cuda:{}'.format(args.device_ids))
    print('Using GPU ID: {}'.format(args.device_ids))

    # set random seed
    set_seed(args.seed)
    print('Using Random Seed: {}'.format(str(args.seed)))

    # set tensorboard
    args.logger_dir = os.path.join(args.logger_dir, args.experiment_name)
    os.makedirs(args.logger_dir, exist_ok=True)
    writer = SummaryWriter(args.logger_dir)
    print('Set Tensorboard: {}'.format(args.logger_dir))

    csv_path = os.path.join(args.csv_dir, 'Fold_{}.csv'.format(args.fold))  # dir to save label
    if args.test:
        train_dataset, val_dataset, test_dataset = return_splits(csv_path=csv_path, test=True)
        args.dataset = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        train_eset, val_eset, test_eset = train_dataset.keys(), val_dataset.keys(), test_dataset.keys()
        args.data_eset = {'train_eset': train_eset, 'val_eset': val_eset, 'test_eset': test_eset}
    else:
        train_dataset, val_dataset = return_splits(csv_path=csv_path, test=False)
        args.dataset = {'train': train_dataset, 'val': val_dataset}
        train_eset, val_eset = train_dataset.keys(), val_dataset.keys()
        args.data_eset = {'train_eset': train_eset, 'val_eset': val_eset}

    feat_dir = args.feat_dir
    train_dset = MILDataset(train_dataset, feat_dir)
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=0)
    val_dset = MILDataset(val_dataset, feat_dir)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=0)
    if args.test:
        test_dset = MILDataset(test_dataset, feat_dir)
        test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    split = None
    split_num = 0
    if args.pseudo_step >= args.max_pseudo_num:
        split_list = [args.max_pseudo_num]
    else:
        split_list = list(range(1, args.max_pseudo_num + 1, args.pseudo_step))
        if split_list[-1] != args.max_pseudo_num:
            split_list.append(args.max_pseudo_num)

    model_dir = os.path.join(args.ckpt_dir, args.experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    for round_id in range(args.rounds):
        if 'ABMIL' == args.MIL_model:
            model = ABMIL(n_classes=args.num_classes)
        elif 'DSMIL' == args.MIL_model:
            model = DSMIL(n_classes=args.num_classes)
        elif 'CLAM_SB' == args.MIL_model:
            model = CLAM_SB(size_arg="small", k_sample=8, n_classes=args.num_classes, instance_loss_fn=criterion)
        elif 'CLAM_MB' == args.MIL_model:
            model = CLAM_MB(size_arg="small", k_sample=8, n_classes=args.num_classes, instance_loss_fn=criterion)
        elif 'MeanMIL' == args.MIL_model:
            model = MeanMIL(n_classes=args.num_classes)
        elif 'MaxMIL' == args.MIL_model:
            model = MaxMIL(n_classes=args.num_classes)
        else:
            raise NotImplementedError
        model = model.to(device)
        lr = args.max_lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        model_path = os.path.join(model_dir, '{}_model_{}.pth'.format(args.MIL_model, round_id))
        pre_model_path = os.path.join(model_dir, '{}_model_{}.pth'.format(args.MIL_model, round_id - 1))
        if args.split_data:
            init_model_path = pre_model_path
            if round_id == 0:
                list_idx = 0
                split_num = split_list[list_idx]
            else:
                list_idx = len(split_list) - 1
                split_num = split_list[list_idx]
        if not os.path.exists(model_path):
            if args.split_data and split_num > 1:
                split = split_slide_data(split_num, model, init_model_path, train_loader, device, args.MIL_model,
                                         split, args.metrics, args.search_rate, args.sample_rate)
            if args.split_data:
                model.reset()
            early_stopping = EarlyStopping(model_path=model_path, patience=args.patience, verbose=True, count_loss=True)
            for epoch in range(args.epochs):
                train_loss, train_acc, train_auc, train_f1, train_mat = MIL_train_epoch(round_id, epoch, model,
                                                                                        optimizer, train_loader,
                                                                                        criterion, device,
                                                                                        args.num_classes,
                                                                                        args.MIL_model, split)
                val_loss, val_acc, val_auc, val_f1, val_mat = MIL_pred(round_id, model, val_loader, criterion,
                                                                       device, args.num_classes, args.MIL_model)
                if args.metric2save == 'acc':
                    counter = early_stopping(epoch, val_loss, model, val_acc)
                elif args.metric2save == 'f1':
                    counter = early_stopping(epoch, val_loss, model, val_f1)
                elif args.metric2save == 'auc':
                    counter = early_stopping(epoch, val_loss, model, val_auc)
                elif args.metric2save == 'acc_auc':
                    counter = early_stopping(epoch, val_loss, model, (val_acc + val_auc) / 2)
                elif args.metric2save == 'f1_auc':
                    counter = early_stopping(epoch, val_loss, model, (val_f1 + val_auc) / 2)
                elif args.metric2save == 'loss':
                    counter = early_stopping(epoch, val_loss, model)
                else:
                    raise NotImplementedError
                if early_stopping.early_stop:
                    if args.split_data:
                        if split_num == split_list[-1]:
                            print('Early Stopping')
                            break
                    else:
                        print('Early Stopping')
                        break
                # adjust learning rate
                if counter == args.lr_patience:
                    if lr == args.min_lr and args.split_data:
                        if split_num != split_list[-1]:
                            early_stopping.reset()
                            list_idx = list_idx + 1 if list_idx < len(split_list) - 1 else list_idx
                            split_num = split_list[list_idx]
                        if split_num > 1:
                            split = split_slide_data(split_num, model, model_path, train_loader, device, args.MIL_model,
                                                     split, args.metrics, args.search_rate, args.sample_rate)
                if counter > 0 and counter % args.lr_patience == 0:
                    if lr > args.min_lr:
                        early_stopping.reset()
                        lr = lr / 10 if lr / 10 >= args.min_lr else args.min_lr
                        for params in optimizer.param_groups:
                            params['lr'] = lr

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        val_loss, val_acc, val_auc, val_f1, val_mat = MIL_pred(round_id, model, val_loader, criterion, device,
                                                               args.num_classes, args.MIL_model)
        draw_metrics(writer, 'Val', args.num_classes, val_loss, val_acc, val_auc, val_mat, val_f1, round_id)
        if args.test:
            test_loss, test_acc, test_auc, test_f1, test_mat = MIL_pred(round_id, model, test_loader, criterion, device,
                                                                        args.num_classes, args.MIL_model, 'Test')
            draw_metrics(writer, 'Test', args.num_classes, test_loss, test_acc, test_auc, test_mat, test_f1, round_id)
