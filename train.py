import argparse
import os
from datetime import time

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import  numpy as np
import  random
# from loss import Loss
from dataloader import load_data
from loss import MetaComprehensiveRegularization as Loss
from loss import shannon_entropy as Loss2
from metric import valid
from network import GraphFilter
from network import Network
import time



parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default='Hdigit')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=3e-4)
parser.add_argument("--fussion_learning_rate", default=3e-4)
parser.add_argument("--learning_rate1", default=3e-4)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=100)
parser.add_argument("--con_epochs", default=200)
parser.add_argument("--tune_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--neighbor_num", default=4)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Dataname = args.dataset
print("Using {} dataset".format(args.dataset))
file = open("result_method.csv", "a+")
print(args.dataset, file=file)
file.close()

if args.dataset == "Hdigit":

    args.mse_epochs = 50
    args.con_epochs = 100
    seed = 5
    lam1 = 1e-8
    lam2 = 1e-7


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(seed)


dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

# warm up
def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, _, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


# meta information bi-level optimization
def comp_regularize(epoch):
    tot_lossin = 0.
    tot_lossou = 0.
    tot_loss1 = 0.
    mes = torch.nn.MSELoss()
    critertion = Loss()

    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()

        xrs, zs, zc = model(xs)
        loss_inner = []
        loss_outer = []
        loss_1 = []
        for v in range(view):
            # Loss rec
            loss_inner.append(mes(xs[v], xrs[v]))
        # Loss cva
        loss1= lam1 * critertion.forward_special(zs, zc)
        loss_inner.append(loss1)
        loss_1.append(loss1)

        loss = sum(loss_inner)
        loss.backward()
        optimizer1.step()
        tot_lossin += loss.item()
        tot_loss1 += sum(loss_1).item()
        optimizer_fussion.zero_grad()

        xrs1, zs1, zc1 = model(xs)
        # Loss mic
        loss_outer.append(lam2 * Loss2(zc1))
        loss2 = sum(loss_outer)

        loss2.backward()
        tot_lossin += loss2.item()
        optimizer_fussion.step()
        tot_lossou +=loss2.item()
    print('Epoch {}'.format(epoch), 'Loss_all:{:.6f}'.format(tot_lossin / len(data_loader)))
    loss_all.append(tot_lossin / len(data_loader))
    print('Epoch {}'.format(epoch), 'Loss_cva:{:.6f}'.format(tot_loss1 / len(data_loader)))
    print('Epoch {}'.format(epoch), 'Loss_mic:{:.6f}'.format(tot_lossou / len(data_loader)))

if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1
k = args.neighbor_num
max_acc=0
max_nmi=0
max_ari=0
max_pur=0
max_f = 0
epoch_list = []
loss_all = []
acc_list = []
nmi_list = []
ari_list = []
pur_list = []
f1_list = []
# T=10
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    # optimizer for warm up
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer for outer-level
    optimizer_fussion = torch.optim.Adam(model.fussion.parameters(), lr=args.fussion_learning_rate, weight_decay=args.weight_decay)
    # optimizer for inner-level
    optimizer1 = torch.optim.Adam(model.parameters(), lr=args.learning_rate1, weight_decay=args.weight_decay)

    # semantic puzzle
    gat = GraphFilter()

    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1

    while epoch <= args.mse_epochs + args.con_epochs:
        comp_regularize(epoch)

        if epoch%5==0 or epoch==args.mse_epochs+1:

            vs = False

            acc, nmi, ari, pur, f = valid(model, device, dataset, view, data_size, class_num, gat, epoch, vs,
                                              eval_h=True)

            acc_list.append(acc)
            nmi_list.append(nmi)
            epoch_list.append(epoch)
            if acc > max_acc:
                max_acc=acc
                max_nmi=nmi
                max_ari=ari
                max_pur=pur
                max_f = f
                max_epoch=epoch

        epoch += 1



    tqdm.write('MAX: acc: {:.4f}, nmi: {:.4f}, ari: {:.4f}, pur: {:.4f},f1: {:.4f}, lam1: {}, lam2: {}, epoch: {}, seed: {}, lr: {}, wp: {}'.format(max_acc, max_nmi, max_ari, max_pur, max_f, lam1, lam2, max_epoch, seed, args.learning_rate, args.mse_epochs))
    file = open("result_method.csv", "a+")
    print('acc: {:.4f}, nmi: {:.4f}, ari: {:.4f}, pur: {:.4f},f1: {:.4f}, lam1: {}, lam2: {}, epoch: {}, seed: {},lr: {}, wp: {}'.format(max_acc, max_nmi, max_ari, max_pur, max_f, lam1, lam2, max_epoch, seed, args.learning_rate, args.mse_epochs), file=file)
    file.close()

