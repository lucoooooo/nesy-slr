from abc import ABC, abstractmethod
import os
import pickle
import random
from zipfile import ZipFile
from typing import Callable, Tuple
import torchvision
from io import BytesIO
from problog.logic import Term, Constant
from torch.utils.data import Dataset as TorchDataset
import random
from typing import *
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset as TorchDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score
from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, Constant
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import TrainObject
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.dataset import DataLoader as DL


mnistbasic_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
    )
])
mnistb0_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.Resize(64, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.1307,)*3,
                        std=(0.3081,)*3),
])

mnistb3_img_transform = mnistb0_img_transform

class MNIST_Images(object):
    def __init__(self, dataset, device):
        self.device = device
        self.dataset = dataset

    def __getitem__(self, item):
        return self.dataset[int(item[0])][0].to(self.device)

class MNISTSum2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        seed : int = None
    ):
        self.dataset_name = "train" if train else "test"
        self.seed = seed
        self.mnist_dataset = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        )
        self.index_map = list(range(len(self.mnist_dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.index_map)
        else:
            random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.mnist_dataset) / 2)

    def __getitem__(self, idx):
        
        (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
        (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]


        return (a_img, b_img, a_digit + b_digit)

    @staticmethod
    def collate_fn(batch):
        a_imgs = torch.stack([item[0] for item in batch])
        b_imgs = torch.stack([item[1] for item in batch])
        digits = torch.stack([torch.tensor(item[2]).long() for item in batch])
        return ((a_imgs, b_imgs), digits)
    
class MNISTSum2Dataset_SYM(Dataset,TorchDataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        seed : int = None
    ):
        self.dataset_name = "train" if train else "test"
        self.seed = seed
        self.mnist_dataset = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        )
        self.index_map = list(range(len(self.mnist_dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.index_map)
        else:
            random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.index_map) / 2)
    
    def __getitem__(self, idx):
        real_idx = idx % (len(self.index_map)//2)
        i1 = self.index_map[real_idx * 2]
        i2 = self.index_map[real_idx * 2 + 1]
        return (i1,i2)

    def to_query(self,idx):
        i1, i2 = self.__getitem__(idx)
        (_, a_digit) = self.mnist_dataset[i1]
        (_, b_digit) = self.mnist_dataset[i2]
        summation = a_digit + b_digit
    
        X=Term("X")
        Y=Term("Y")
        subs = {
            X : Term("tensor", Term(self.dataset_name, Constant(i1))),
            Y : Term("tensor", Term(self.dataset_name, Constant(i2)))
        }

        x = Query(
            Term("sum2", X, Y, Constant(summation)),
            substitution=subs
        )
        return x

    def to_queries(self):
        return [self.to_query(i) for i in range(len(self))]

def createSYMdataset(trainset : MNISTSum2Dataset, testset : MNISTSum2Dataset):
    
    trainb = clone_with_transform(trainset, mnistbasic_img_transform, True)
    testb = clone_with_transform(testset, mnistbasic_img_transform, True)
    trainb0 = clone_with_transform(trainset, mnistb0_img_transform, True)
    testb0 = clone_with_transform(testset, mnistb0_img_transform, True)
    trainb3 = clone_with_transform(trainset, mnistb3_img_transform, True)
    testb3 = clone_with_transform(testset, mnistb3_img_transform, True)

    return {
        "basic": {
            "trainset": trainb,
            "testset": testb
        },
        "b0": {
            "trainset": trainb0,
            "testset": testb0
        },
        "b3": {
            "trainset": trainb3,
            "testset": testb3
        }  
    }

def clone_with_transform(dataset, transform, is_sym : bool = False):
    dataset_type = MNISTSum2Dataset_SYM if is_sym else MNISTSum2Dataset
    clone = dataset_type(
            root = dataset.mnist_dataset.root,
            train = True if dataset.dataset_name == "train" else False,
            transform = transform,
            download = False,
            seed = dataset.seed
        )
    if hasattr(dataset, "index_map"):
        clone.index_map = dataset.index_map
    return clone


def mnist_sum_2_loader(
    batch_size_train,
    batch_size_test,
    modeltype: str,
    train_dataset: MNISTSum2Dataset,
    test_dataset: MNISTSum2Dataset,
):
    if modeltype == "mnistb0":
        mnist_img_transform = mnistb0_img_transform
    elif modeltype == "mnistb3":
        mnist_img_transform = mnistb3_img_transform
    else:
        mnist_img_transform = mnistbasic_img_transform

    train_ds = clone_with_transform(train_dataset, mnist_img_transform)

    test_ds = clone_with_transform(test_dataset, mnist_img_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=False
    )
    return train_loader, test_loader

class MNISTNet(nn.Module, ABC):
    def __init__(self):
        super(MNISTNet, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

class MNISTNet_basic(nn.Module):
    def __init__(self):
        super(MNISTNet_basic, self).__init__()
        self.modelname = "basic"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)  
        self.fc1 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        #x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        #x = F.relu(x)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        x = self.bn3(x) 
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.softmax(x,dim=1) 


class MNISTNet_b0(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b0, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.modelname = "b0"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x):   
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        return F.softmax(x,dim=1)

class MNISTNet_b3(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b3, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.modelname = "b3"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x):   
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        return F.softmax(x,dim=1)

    
class MNISTSum2Net(nn.Module):
    def __init__(self, model):
        super(MNISTSum2Net, self).__init__()
        self.mnist_net = model


    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        (a_imgs, b_imgs) = x

        # First recognize the two digits
        a_distrs = self.mnist_net(a_imgs)
        b_distrs = self.mnist_net(b_imgs) 

        return (a_distrs, b_distrs) 
    
def conv_sum(d1, d2):  
    B = d1.size(0)
    x = d1.unsqueeze(0)                     
    w = d2.unsqueeze(1).flip(2)             
    y = nn.functional.conv1d(x, w, groups=B, padding=9)  
    p_sum = y.squeeze(0)[:, :19]             
    p_sum = p_sum / (p_sum.sum(dim=1, keepdim=True) + 1e-12)
    return p_sum
    
class Trainer_NoSym:
    def __init__(self, train_loader, test_loader, model_dir, learning_rate, model : MNISTSum2Net, device):
        self.model_dir = model_dir
        self.network = model
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = nn.NLLLoss()
        self.device = device
        self.network.to(device)

    def train(self, num_epoch):
        train_losses, acc_train = [], []
        for epoch in range(num_epoch):
            y_true, y_pred = [], []
            running_loss = 0.0
            self.network.train()
            for ((d1,d2), target) in tqdm(self.train_loader, total=len(self.train_loader), desc='Train Loop'):
                self.optimizer.zero_grad()
                d1,d2 = d1.to(self.device), d2.to(self.device)
                target = target.to(self.device)
                (dprob1, dprob2) = self.network((d1,d2))
                output = conv_sum(F.softmax(dprob1, dim=1), F.softmax(dprob2, dim=1))
                loss = self.loss(torch.log(output + 1e-9), target.long())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * target.size(0) 
                preds = output.argmax(dim=1)
                y_true.extend(target.cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())
            
            train_loss = running_loss / len(self.train_loader.dataset)
            train_losses.append(train_loss)
            running_train_accuracy = accuracy_score(y_true, y_pred)
            acc_train.append(running_train_accuracy)
            
            #stats
            print(f"Epoca {epoch+1}/{num_epoch} - Train loss: {train_loss} with accuracy: {running_train_accuracy*100}%")
        
        plt.plot(train_losses, label='Training loss')
        plt.legend()
        plt.title("Loss over epochs")
        plt.show()
        plt.plot(acc_train, label='Accuracy in training')
        plt.legend()
        plt.title("Accuracy over epochs")
        plt.show()
        torch.save(self.network.state_dict(), os.path.join(self.model_dir, f"{self.network.mnist_net.modelname}_nosym.pt"))
        return {
            "loss": min(train_losses),
            "accuracy" : max(acc_train),
        }
    
    def test(self):
        self.network.eval()
        y_true, y_pred = [], []
        running_loss = 0.0
        with torch.no_grad():
            for ((d1,d2), target) in tqdm(self.test_loader, total=len(self.test_loader), desc='Test Loop'):
                d1,d2 = d1.to(self.device), d2.to(self.device)
                target = target.to(self.device)
                (dprob1, dprob2) = self.network((d1,d2))
                output = conv_sum(F.softmax(dprob1, dim=1), F.softmax(dprob2, dim=1))
                running_loss += self.loss(torch.log(output + 1e-9), target.long()).item() * target.size(0)
                preds = output.argmax(dim=1)
                y_true.extend(target.cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

        test_loss = running_loss / len(self.test_loader.dataset)
        acc = accuracy_score(y_true, y_pred)
        #cm = confusion_matrix(y_true, y_pred)
        return {
            "loss": test_loss,
            "accuracy" : acc,
        }
    
class DeepProblogModel:
    def __init__(self, net: MNISTNet, modeldir, datadir,device, trainset : MNISTSum2Dataset_SYM, testset : MNISTSum2Dataset_SYM, batch_size_train, learning_rate, seed, cache = False):
        self.net = Network(net, "mnist_net", batching=True)
        self.device = device
        self.net.network_module.to(self.device)
        self.MNIST_train = MNIST_Images(trainset.mnist_dataset, self.device)
        self.MNIST_test = MNIST_Images(testset.mnist_dataset, self.device)
        self.trainset = trainset
        self.testset = testset
        self.batch_size_train = batch_size_train
        self.net.device = self.device
        self.net.optimizer = torch.optim.Adam(self.net.network_module.parameters(), learning_rate)
        self.model = Model("./model/sum2.pl", [self.net])
        self.model.set_engine(ExactEngine(self.model), cache=cache)
        self.model.add_tensor_source("train", self.MNIST_train)
        self.model.add_tensor_source("test", self.MNIST_test)
        self.seed = seed
        self.train_object = TrainObject(self.model)
        self.train_loader = DL(self.trainset, self.batch_size_train, seed = self.seed)
        self.datadir = datadir
        self.modeldir = modeldir

    def train(self, n_epochs):
        self.train_object.train(self.train_loader, n_epochs)
        self.model.save_state(self.modeldir+f"/{self.net.network_module.modelname}_sym.pth")
        self.saveResults(True)
        

    def test(self):
        cm = get_confusion_matrix(self.model, self.testset, verbose=1)
        self.train_object.logger.comment(
            "Test Accuracy {}".format(cm.accuracy())
        )
        self.train_object.logger.comment(
            "TEST Confusion Matrix {}".format(cm.__str__())
        )
        self.saveResults()


    #override della funzione in network.py e model.py in modo tale da poter passare da pth fatti con cuda a cpu
    def load_state_dict(self, path : Union[str, os.PathLike, IO[bytes]], device):
        with ZipFile(path) as zipf:
            with zipf.open("parameters") as f:
                self.model.parameters = pickle.load(f)
            for n in self.model.networks:
                with zipf.open(n) as f:
                    location = BytesIO(f.read())
                    state :  Dict[str, Any] = torch.load(location, map_location=device)
                    self.model.networks[n].network_module.load_state_dict(state["model_state_dict"])
                    if "optimizer_state_dict" in state:
                        assert self.net.optimizer is not None
                        self.model.networks[n].optimizer.load_state_dict(state["optimizer_state_dict"])
                    if "scheduler_state_dict" in state:
                        assert self.net.scheduler is not None
                        self.model.networks[n].scheduler.load_state_dict(state["scheduler_state_dict"])
        

    def saveResults(self,train = False):
        phase = "train" if train else "test"
        self.train_object.logger.write_to_file(self.data_dir+f"/{self.net.network_module.modelname}_{phase}_sym")