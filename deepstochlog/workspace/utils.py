from abc import ABC, abstractmethod
import os
import pickle
import random
from zipfile import ZipFile
from typing import Callable, Tuple, Optional, Union, Sequence
from io import BytesIO
from torch.utils.data import Dataset as TorchDataset
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import timm
import time
from sklearn.metrics import accuracy_score
from deepstochlog.utils import create_model_accuracy_calculator
from deepstochlog.network import Network, NetworkStore
from deepstochlog.model import DeepStochLogModel
from deepstochlog.term import Term, List
from deepstochlog.trainer import DeepStochLogTrainer, print_logger
from deepstochlog.dataloader import DataLoader as DL
from deepstochlog.context import Context, ContextualizedTerm
import gc

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
    #torchvision.transforms.Normalize(mean=(0.1307,)*3,
    #                    std=(0.3081,)*3),
    torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],
                                    std=[0.229,0.224,0.225]),
])

mnistb3_img_transform = mnistb0_img_transform

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


        return (a_img, b_img), (a_digit, b_digit)

    @staticmethod
    def collate_fn(batch):
        a_imgs = torch.stack([item[0][0] for item in batch])
        b_imgs = torch.stack([item[0][1] for item in batch])
        a_digits = torch.tensor([item[1][0] for item in batch])
        b_digits = torch.tensor([item[1][1] for item in batch])
        return ((a_imgs, b_imgs), (a_digits,b_digits))
    

class MNISTSum2Dataset_SYM(Sequence):
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
        self.A=Term("a")
        self.B=Term("b")
        self.argument_seq = List(self.A,self.B)
        self.index_map = list(range(len(self.mnist_dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.index_map)
        else:
            random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.index_map) // 2)
    
    def __getindex__(self, idx):
        i1 = self.index_map[idx * 2]
        i2 = self.index_map[idx * 2 + 1]
        return (i1,i2)

    def to_query(self,idx):
        i1, i2 = self.__getindex__(idx)
        (a_img, a_digit) = self.mnist_dataset[i1]
        (b_img, b_digit) = self.mnist_dataset[i2]
        summation = a_digit + b_digit
    
        query = ContextualizedTerm(
            context = Context({self.A: a_img, self.B: b_img}),
            term = Term(
                "addition",
                Term(str(summation)),
                self.argument_seq,
            ),
            meta = str(a_digit) + "+" + str(b_digit)
        )

        return query

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, slice):
            return (self.to_query(i) for i in range(*item.indices(len(self))))
        return self.to_query(item)

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

class MNISTNet_basic(MNISTNet):
    def __init__(self, num_classes=10):
        super(MNISTNet_basic, self).__init__()
        self.modelname = "basic"
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, **kwargs):
        neural = kwargs.get('neural', False)
        x = self.conv1(x)
        x = F.relu(x)       
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)    
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        if neural is True:
            return x
        else:
            return F.softmax(x,dim=1)


class MNISTNet_b0(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b0, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.modelname = "b0"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x,**kwargs):
        neural = kwargs.get('neural', False)
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        if neural is True:
            return x
        else:
            return F.softmax(x,dim=1)

class MNISTNet_b3(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b3, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.modelname = "b3"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x,**kwargs):
        neural = kwargs.get('neural', False)
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        if neural is True:
            return x
        else:
            return F.softmax(x,dim=1)

    
class MNISTSum2Net(nn.Module):
    def __init__(self, model):
        super(MNISTSum2Net, self).__init__()
        self.mnist_net = model
        self.feature_dim = 10

        combined_dim = self.feature_dim * 2
        self.sum_classifier = nn.Linear(combined_dim, 19)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        (a_imgs, b_imgs) = x

        a_feat = self.mnist_net(a_imgs, neural=True)
        b_feat = self.mnist_net(b_imgs, neural=True)
        
        a_pred = (F.softmax(a_feat, dim=1)).argmax(dim=1)
        b_pred = (F.softmax(b_feat, dim=1)).argmax(dim=1)

        combined_feature = torch.cat((a_feat,b_feat), dim=1)

        sumpred = F.softmax(self.sum_classifier(combined_feature), dim=1)

        return sumpred, a_pred, b_pred

    

class Trainer_NoSym:
    def __init__(self, train_loader, test_loader, model_dir,learning_rate, model : MNISTSum2Net, device):
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
        d1_acc_train, d2_acc_train = [], []
        for epoch in range(num_epoch):
            y_true, y_pred = [], []
            a_true, a_pred = [], []
            b_true, b_pred = [], []
            running_loss = 0.0
            self.network.train()
            for ((img1,img2),(d1,d2)) in tqdm(self.train_loader, total=len(self.train_loader), desc='Train Loop'):
                self.optimizer.zero_grad()
                img1,img2 = img1.to(self.device), img2.to(self.device)
                target = d1 + d2
                target = target.to(self.device)
                output, d1pred, d2pred= self.network((img1,img2))
                loss = self.loss(torch.log(output + 1e-9), target.long())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * target.size(0) 
                preds = output.argmax(dim=1)

                y_true.extend(target.cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

                a_true.extend(d1.cpu().tolist())
                a_pred.extend(d1pred.detach().cpu().tolist())

                b_true.extend(d2.cpu().tolist())
                b_pred.extend(d2pred.detach().cpu().tolist())
            
            train_loss = running_loss / len(self.train_loader.dataset)
            train_losses.append(train_loss)
            running_train_accuracy = accuracy_score(y_true, y_pred)
            d1acc = accuracy_score(a_true, a_pred)
            d2acc = accuracy_score(b_true, b_pred)
            d1_acc_train.append(d1acc)
            d2_acc_train.append(d2acc)
            acc_train.append(running_train_accuracy)
            
            #stats
            print(f"Epoca {epoch+1}/{num_epoch} - Train loss: {train_loss} with total accuracy: {running_train_accuracy*100}% \n digit1 accuracy: {d1acc} and digit2 accuracy: {d2acc}")
        
        torch.save(self.network.state_dict(), os.path.join(self.model_dir, f"{self.network.mnist_net.modelname}_neural.pt"))
        return {
            "loss": train_losses,
            "accuracy" : acc_train,
            "single-digit accuracy": (max(d1_acc_train), max(d2_acc_train))
        }
    
    def test(self):
        self.network.eval()
        y_true, y_pred = [], []
        a_true, a_pred = [], []
        b_true, b_pred = [], []
        running_loss = 0.0
        with torch.no_grad():
            for ((img1,img2),(d1,d2)) in tqdm(self.test_loader, total=len(self.test_loader), desc='Test Loop'):
                img1,img2 = img1.to(self.device), img2.to(self.device)
                target = d1 + d2
                target = target.to(self.device)
                output, d1pred, d2pred = self.network((img1,img2))
                running_loss += self.loss(torch.log(output + 1e-9), target.long()).item() * target.size(0)
                preds = output.argmax(dim=1)
                
                y_true.extend(target.cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

                a_true.extend(d1.cpu().tolist())
                a_pred.extend(d1pred.detach().cpu().tolist())

                b_true.extend(d2.cpu().tolist())
                b_pred.extend(d2pred.detach().cpu().tolist())

        test_loss = running_loss / len(self.test_loader.dataset)
        acc = accuracy_score(y_true, y_pred)
        d1acc = accuracy_score(a_true,a_pred)
        d2acc = accuracy_score(b_true,b_pred)
        

        print(f"Test loss: {test_loss} with total accuracy: {acc*100}% \n digit1 accuracy: {d1acc} and digit2 accuracy: {d2acc}")
        return {
            "loss": test_loss,
            "accuracy" : acc,
            "single-digit accuracy": (d1acc,d2acc)
        }

class Trainer_Sym:
    def __init__(self, net: MNISTNet, modeldir, datadir,device, trainset : MNISTSum2Dataset_SYM, testset : MNISTSum2Dataset_SYM, batch_size_train,batch_size_test, learning_rate, onlyTest=False, max_size = 15000):
        self.mnist_classifier = net
        self.net = Network("mnist", self.mnist_classifier, index_list = [Term(str(i)) for i in range(10)])
        self.device = device
        self.nets = NetworkStore(self.net)
        query = Term(
            "addition",
            Term("_"),
            List(Term("a"), Term("b")),
        )   
        self.model = DeepStochLogModel.from_file(
            file_location = os.path.abspath("./model/sum2.pl"),
            query = query,
            networks = self.nets,
            device = self.device,
            verbose = True
        )
        self.singledigittrain = torch.utils.data.DataLoader(
            trainset.mnist_dataset,
            batch_size=batch_size_train,
            shuffle=True
        )
        self.singledigittest = torch.utils.data.DataLoader(
            testset.mnist_dataset,
            batch_size=batch_size_test,
            shuffle=True
        )
        self.optimizer = optim.Adam(self.model.get_all_net_parameters(), lr=learning_rate)
        if not onlyTest:
            self.trainloader = DL(trainset, batch_size=batch_size_train, max_size=max_size)
            self.trainloader_test = DL(trainset, batch_size=batch_size_train, max_size=max_size)
            compute_accuracy = create_model_accuracy_calculator(
                    self.model,
                    self.trainloader_test,
                    time.time(),
            )
            self.trainer = DeepStochLogTrainer(
                    log_freq=100,
                    accuracy_tester = compute_accuracy,
                    logger=print_logger,
                    print_time=True
            )
        self.testloader = DL(testset, batch_size=batch_size_test)
        _, self.tester = create_model_accuracy_calculator(
            self.model,
            self.testloader,
            time.time(),
        )
        self.datadir = datadir
        self.modeldir = modeldir

    def single_digit_acc(self,model, test_loader, device):
        model.eval()
        preds, trues= [], []
        model.to(device)
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                trues.extend(target.cpu().tolist())
                preds.extend(pred.detach().cpu().tolist())
        
        sd_acc = accuracy_score(trues,preds)
        return sd_acc

    def train(self, n_epochs):
	
        _, train_losses, result_acc=self.trainer.train(
            model=self.model,
            optimizer=self.optimizer,
            dataloader=self.trainloader,
            epochs=n_epochs,
        )
        
        torch.save(self.mnist_classifier.state_dict(), os.path.join(self.modeldir, f"{self.mnist_classifier.modelname}_nesy.pt"))
	
        sd_accuracy = self.single_digit_acc(self.mnist_classifier,self.singledigittrain,self.device)
        
        #pulisco ram 
        del self.trainloader
        del self.trainloader_test
        gc.collect()

        train_acc = []
        for acc in result_acc:
            acc = acc.split()
            train_acc.append(float(acc[0]))

        return {
            "loss": train_losses,
            "accuracy": train_acc,
            "single-digit accuracy": (sd_accuracy, sd_accuracy)
        }

    def test(self):
        results = self.tester().split()
        test_acc = results[0]
        sd = self.single_digit_acc(self.mnist_classifier, self.singledigittest, self.device)
        
        #pulisco ram
        del self.testloader
        gc.collect()

        return {
            "loss": None,
            "accuracy": float(test_acc),
            "single-digit accuracy": (sd, sd)
        }

        
