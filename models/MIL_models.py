import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MeanMIL(nn.Module):
    def __init__(self, n_classes=1, dropout=True, act='relu'):
        super(MeanMIL, self).__init__()
        head = [nn.Linear(1024, 512)]
        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]
        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(512, n_classes)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def reset(self):
        initialize_weights(self)

    def forward(self, x):
        x = self.head(x).mean(axis=1)
        return x


class MaxMIL(nn.Module):
    def __init__(self, n_classes=1, dropout=True, act='relu'):
        super(MaxMIL, self).__init__()
        head = [nn.Linear(1024, 512)]
        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]
        if dropout:
            head += [nn.Dropout(0.25)]
        head += [nn.Linear(512, n_classes)]
        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def reset(self):
        initialize_weights(self)

    def forward(self, x):
        x, _ = self.head(x).max(axis=1)
        return x


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class DSMIL(nn.Module):
    def __init__(self, n_classes=2, num_feats=1024):
        super(DSMIL, self).__init__()
        self.i_classifier = FCLayer(num_feats, n_classes)
        self.b_classifier = BClassifier(input_size=num_feats, output_class=n_classes)
        initialize_weights(self)

    def forward(self, x):
        x = x.squeeze(0)
        feats, classes = self.i_classifier(x)
        logit, A, B = self.b_classifier(feats, classes)

        return classes, logit, A, B


class ABMIL(nn.Module):
    def __init__(self, n_classes=2, act='relu', bias=False, dropout=False):
        super(ABMIL, self).__init__()
        self.L = 512
        self.D = 128  # 128
        self.K = 1

        self.feature = [nn.Linear(1024, 512)]
        self.feature += [nn.ReLU()]
        self.feature += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*self.feature)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, n_classes),
        )

        self.attention_a = [
            nn.Linear(self.L, self.D, bias=bias),
        ]
        if act == 'gelu':
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D, bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

        self.apply(initialize_weights)

    def reset(self):
        initialize_weights(self)

    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)

        Y_prob = self.classifier(x)

        return Y_prob, A.detach().cpu().numpy()


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Attn_Net(nn.Module):
    def __init__(self, L=512, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]
        if dropout > 0:
            self.module.append(nn.Dropout(dropout))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x, only_A=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        if only_A:
            return A
        return A, x


class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=4, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def reset(self):
        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A) == 1:
            A = A.squeeze(0)
        sample = max(self.k_sample, 1) if len(A) / 2 >= self.k_sample else max(int(len(A) / 2), 1)
        _, top_p_ids = torch.topk(A, sample, dim=0)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        _, top_n_ids = torch.topk(-A, sample, dim=0)
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(sample, device)
        n_targets = self.create_negative_targets(sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A) == 1:
            A = torch.squeeze(A)
        sample = max(self.k_sample, 1) if len(A) / 2 >= self.k_sample else max(int(len(A) / 2), 1)
        _, top_p_ids = torch.topk(A, sample, dim=0)
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        h = torch.squeeze(h)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = A.view(1, -1)
        if attention_only:
            return A
        A = F.softmax(A, dim=1)  # softmax over N
        A_raw = A.detach().cpu().numpy()

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': h.detach()})
        return logits, A_raw, results_dict


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=4, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False):
        nn.Module.__init__(self)
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        h = torch.squeeze(h)
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A.detach().cpu().numpy()
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, A_raw, results_dict

