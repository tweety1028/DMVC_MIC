from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels


    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        self.y = scipy.io.loadmat(path+'CCV.mat')['Y'].astype(np.int32)
        # self.data1 = scaler.fit_transform(self.data1)
        self.x1 = scaler.fit_transform(scipy.io.loadmat(path+'CCV.mat')['X'][0][0].astype(np.float32))
        self.x2 = scaler.fit_transform(scipy.io.loadmat(path+'CCV.mat')['X'][0][1].astype(np.float32))
        self.x3 = scaler.fit_transform(scipy.io.loadmat(path+'CCV.mat')['X'][0][2].astype(np.float32))
        print(self.x1.shape)
        # self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        # x1 = self.data1[idx]
        # x2 = self.data2[idx]
        # x3 = self.data3[idx]

        return [torch.from_numpy(self.x1), torch.from_numpy(
           self.x2), torch.from_numpy(self.x3)], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Hdigit(Dataset):
        def __init__(self, path):
            self.y= scipy.io.loadmat(path + 'Hdigit.mat')['truelabel'][0][0].T.astype(np.int32)
            self.x1 = scipy.io.loadmat(path + 'Hdigit.mat')['data'][0][0].T.astype(np.float32)
            self.x2 = scipy.io.loadmat(path + 'Hdigit.mat')['data'][0][1].T.astype(np.float32)
            # print(11)
            # print(self.y.shape)

        def __len__(self):
            return self.x1.shape[0]

        def __getitem__(self, idx):
            return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
                self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class ALOI(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        self.y = scipy.io.loadmat(path + 'ALOI-100.mat')['gt'].astype(np.int32)
        self.x1 = scaler.fit_transform(scipy.io.loadmat(path + 'ALOI-100.mat')['fea'][0][0].astype(np.float32))
        # print(self.x1.shape)
        self.x2 = scaler.fit_transform(scipy.io.loadmat(path + 'ALOI-100.mat')['fea'][0][1].astype(np.float32))
        self.x3 = scaler.fit_transform(scipy.io.loadmat(path + 'ALOI-100.mat')['fea'][0][2].astype(np.float32))
        self.x4 = scaler.fit_transform(scipy.io.loadmat(path + 'ALOI-100.mat')['fea'][0][3].astype(np.float32))
        # print(11)
        # print(self.y.shape)

    def __len__(self):
        # print(self.x1.shape[0])
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx]),torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Cora(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        self.y = scipy.io.loadmat(path + 'Cora.mat')['Y'].astype(np.int32)
        self.x1 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][0].astype(np.float32))
        # print(self.x1.shape)
        self.x2 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][1].astype(np.float32))
        self.x3 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][2].astype(np.float32))
        self.x4 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][3].astype(np.float32))
        # print(11)
        # print(self.y.shape)

    def __len__(self):
        # print(self.x1.shape[0])
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx]),torch.from_numpy(self.x3[idx]), torch.from_numpy(self.x4[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class HW(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        self.y = scipy.io.loadmat(path + 'handwritten.mat')['Y'].astype(np.int32)
        self.x1 = scaler.fit_transform(scipy.io.loadmat(path + 'handwritten.mat')['X'][0][0].astype(np.float32))
        # print(self.x1.shape)
        self.x2 = scaler.fit_transform(scipy.io.loadmat(path + 'handwritten.mat')['X'][0][2].astype(np.float32))
        # self.x3 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][2].astype(np.float32))
        # self.x4 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][3].astype(np.float32))
        # print(11)
        # print(self.y.shape)

    def __len__(self):
        # print(self.x1.shape[0])
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
class NoisyMNIST(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        self.y = scipy.io.loadmat(path + 'NoisyMNIST.mat')['trainLabel'].astype(np.int32)
        self.x1 = scaler.fit_transform(scipy.io.loadmat(path + 'NoisyMNIST.mat')['X1'].astype(np.float32))
        # print(self.x1.shape)
        self.x2 = scaler.fit_transform(scipy.io.loadmat(path + 'NoisyMNIST.mat')['X2'].astype(np.float32))
        # self.x3 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][2].astype(np.float32))
        # self.x4 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][3].astype(np.float32))
        # print(11)
        # print(self.y.shape)

    def __len__(self):
        # print(self.x1.shape[0])
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
class WiKi(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        self.y = scipy.io.loadmat(path + 'WiKi.mat')['Y'].astype(np.int32)
        self.x1 = scaler.fit_transform(scipy.io.loadmat(path + 'WiKi.mat')['X'][0][0].astype(np.float32))
        # print(self.x1.shape)
        self.x2 = scaler.fit_transform(scipy.io.loadmat(path + 'WiKi.mat')['X'][0][1].astype(np.float32))
        # self.x3 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][2].astype(np.float32))
        # self.x4 = scaler.fit_transform(scipy.io.loadmat(path + 'Cora.mat')['X'][0][3].astype(np.float32))
        # print(11)
        # print(self.y.shape)

    def __len__(self):
        # print(self.x1.shape[0])
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class RGB(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        self.y = scipy.io.loadmat(path + 'RGB-D.mat')['Y'].transpose()
        self.x1 = scaler.fit_transform(scipy.io.loadmat(path + 'RGB-D.mat')['X1'].astype(np.float32))
        # print(self.x1.shape)
        self.x2 = scaler.fit_transform(scipy.io.loadmat(path + 'RGB-D.mat')['X2'].astype(np.float32))
        # print(11)
        print(self.y.shape)

    def __len__(self):
        # print(self.x1.shape[0])
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784, 256]
        view = 2
        class_num = 10
        data_size = 10000
    elif dataset == "handwritten":
        dataset = HW('./data/')
        dims = [240, 216]
        view = 2
        class_num = 10
        data_size = 2000
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 50000
    elif dataset == "WiKi":
        dataset = WiKi('./data/')
        dims = [128, 10]
        view = 2
        class_num = 10
        data_size = 2866
    elif dataset == "ALOI-100":
        dataset = ALOI('./data/')
        dims = [77, 13, 64, 125]
        view = 4
        class_num = 100
        data_size = 10800
    elif dataset == "RGB-D":
        dataset = RGB('./data/')
        dims = [2048, 300]
        view = 2
        data_size = 1449
        class_num = 13
    elif dataset == "Cora":
        dataset = Cora('./data/')
        dims = [2708, 1433, 2708, 2708]
        view = 4
        data_size = 2708
        class_num = 7
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
