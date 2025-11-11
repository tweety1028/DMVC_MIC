from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from kmeans_gpu import kmeans as KMeansGpu
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from munkres import Munkres

from sklearn import metrics
# from network import vis


def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc, f = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur, f


def inference(loader, model, device, view, data_size, gat):

    model.eval()

    Zs = []
    Zc = []
    Zf = []
    for v in range(view):
        Zs.append([])
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():

            _, zs, zc = model.forward(xs)

            zf = gat(zc)
            zc = zc.detach()
        for v in range(view):

            zs[v] = zs[v].detach()

            Zs[v].extend(zs[v].cpu().detach().numpy())

        Zf.extend(zf.cpu().detach().numpy())
        Zc.extend(zc.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)

    for v in range(view):

        Zs[v] = np.array(Zs[v])

    Zc = np.array(Zc)
    ZF = np.array(Zf)
    return  labels_vector, Zs, Zc, ZF


def valid(model, device, dataset, view, data_size, class_num,gat,epoch,vs,eval_h=False):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )

    labels_vector, low_level_vectors, Zc, Zf = inference(test_loader, model, device, view, data_size, gat)

    if eval_h:


        aggregated_features_c = Zf

        y_pred_all, _ = KMeansGpu(X=torch.from_numpy(aggregated_features_c), num_clusters=class_num, distance='euclidean',
                                tol=1e-4, device=torch.device('cuda'))


        nmi3, ari3, acc3, pur3, f3 = evaluate(labels_vector, y_pred_all.cpu().numpy())


    return acc3, nmi3, ari3, pur3, f3
