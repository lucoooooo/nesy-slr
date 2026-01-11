import os
import random
from typing import *
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
from neurasp import NeurASP
import time

def time_delta_now(t_start: float, simple_format=False) -> str:
    a = t_start
    b = time.time() 
    c = b - a  
    days = round(c // 86400)
    hours = round(c // 3600 % 24)
    minutes = round(c // 60 % 60)
    seconds = round(c % 60)
    millisecs = round(c % 1 * 1000)
    if simple_format:
        return f"{hours}h:{minutes}m:{seconds}s"

    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds", c



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
        # Contains a MNIST dataset
        self.mnist_dataset = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        )
        self.dataset_name = "train" if train else "test"
        self.seed=seed
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


class MNISTSum2NeurASP(torch.utils.data.Dataset):
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
        (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
        (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]
     
        return a_img, a_digit, b_img, b_digit
        
def clone_with_transform(base_ds: MNISTSum2Dataset, data_dir, train_flag: bool, transform, sym=False):
        if not sym:
            ds = MNISTSum2Dataset(
                data_dir, train=train_flag, download=True, transform=transform, seed = base_ds.seed
            )
        else:
            ds = MNISTSum2NeurASP(data_dir, train=train_flag, download=True, transform=transform, seed = base_ds.seed)
        ds.index_map = list(base_ds.index_map)
        return ds

def mnist_sum_2_loader(
    data_dir,
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


    train_ds_ns = clone_with_transform(train_dataset, data_dir,True, mnist_img_transform)

    test_ds_ns= clone_with_transform(test_dataset, data_dir,False, mnist_img_transform)
    
    train_ds_s = clone_with_transform(train_dataset, data_dir,True, mnist_img_transform, True)


    train_loader_ns = torch.utils.data.DataLoader(
        train_ds_ns,
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True
    )
    test_loader_ns = torch.utils.data.DataLoader(
        test_ds_ns,
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=False
    )
    return train_loader_ns, test_loader_ns, train_ds_s

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

def replace_bn_with_gn(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups, child.num_features)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)    

class MNISTNet_b0_sym(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b0_sym, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        replace_bn_with_gn(self.model)
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

class MNISTNet_b3_sym(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b3_sym, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        replace_bn_with_gn(self.model)
        self.modelname = "b3"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x, **kwargs):
        neural = kwargs.get('neural', False)
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
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

    def forward(self, x, **kwargs):
        neural = kwargs.get('neural',False)
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

    def forward(self, x, **kwargs):
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

        # First recognize the two digits
        a_feat = self.mnist_net(a_imgs, neural=True)
        b_feat = self.mnist_net(b_imgs, neural=True)

        a_pred = (F.softmax(a_feat,dim=1)).argmax(dim=1)
        b_pred = (F.softmax(b_feat,dim=1)).argmax(dim=1)

        combined_feature = torch.cat((a_feat,b_feat), dim=1)

        sumpred = F.softmax(self.sum_classifier(combined_feature), dim=1)
        return sumpred, a_pred, b_pred

def conv_sum(d1, d2):  
    B = d1.size(0)
    x = d1.unsqueeze(0)                     
    w = d2.unsqueeze(1).flip(2)             
    y = nn.functional.conv1d(x, w, groups=B, padding=9)  
    p_sum = y.squeeze(0)[:, :19]             
    p_sum = p_sum / (p_sum.sum(dim=1, keepdim=True) + 1e-12)
    return p_sum

class Trainer_Sym():
    def __init__(self, train_loader_nesy,train_loader_ns, test_loader_ns, model_dir, learning_rate,batch_size, model : MNISTNet, method, device):
        self.model_dir = model_dir
        self.network = model
        self.device = device
        self.train_set_nesy = train_loader_nesy #dataloader con query per neurasp (mnist sum dataset)
        self.dataList = []
        self.obsList = []
        self.batchsize = batch_size / 2
        for i, (img1,d1,img2,d2) in enumerate(self.train_set_nesy):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            d1 = (torch.tensor(d1)).to(self.device)
            d2 = (torch.tensor(d2)).to(self.device)
            if i >= 5000:
                break
            self.dataList.append({'i1': (img1.unsqueeze(0), {'digit': d1}), 'i2': (img2.unsqueeze(0), {'digit': d2})})
            self.obsList.append(':- not addition(i1, i2, {}).'.format(d1+d2))

        self.train_loader_ns = train_loader_ns #dataloader mnist sum dataset normale (per valutazioni train)
        self.test_loader_ns = test_loader_ns #dataloader mnist sum dataset normale (per testing)
        self.loss = nn.NLLLoss()
        self.method = method
        sum_program = '''
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X),[0,1,2,3,4,5,6,7,8,9]) :- img(X).
'''
        nnMap = {'digit' : self.network}
        optimizer = {'digit' : optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-7)}
        self.nesy = NeurASP(sum_program, nnMap, optimizer, gpu = True if torch.cuda.is_available() else False)       

    def train(self, num_epoch): 
        train_losses, acc_train = [], []
        sd1_accuracy, sd2_accuracy = [], []
    
        train_times = []
        for epoch in range(num_epoch):
            time_train = time.time()
            self.nesy.learn(dataList = self.dataList,obsList = self.obsList, epoch=1,alpha = 0.1, method = self.method,batchSize=self.batchsize,bar=True)
            timestamp_train = time_delta_now(time_train, simple_format=True)
            results_training = self.test(self.train_loader_ns, desc="Evaluating Training Epoch")
            loss = results_training["loss"]
            train_accuracy = results_training["accuracy"]
            d1acc = results_training["single-digit accuracy"][0]
            d2acc = results_training["single-digit accuracy"][1]
            train_losses.append(loss)
            acc_train.append(train_accuracy)
            sd1_accuracy.append(d1acc)
            sd2_accuracy.append(d2acc)
            train_times.append(timestamp_train)
            #stats
            print(f"Epoca {epoch+1}/{num_epoch} - {self.network.modelname} nesy model - Train loss: {loss} with total accuracy: {train_accuracy*100}% \n digit1 accuracy: {d1acc} and digit2 accuracy: {d2acc}")
        
        torch.save(self.network.state_dict(), os.path.join(self.model_dir, f"{self.network.modelname}_nesy.pt"))
        return {
            "loss": train_losses,
            "accuracy" : acc_train,
            "single-digit accuracy": (max(sd1_accuracy), max(sd2_accuracy))
        }
        

    def test(self, dataloader = None, desc="Test Loop"):
        dataloader = self.test_loader_ns if dataloader is None else dataloader
        self.network.eval()
        y_true, y_pred = [], []
        a_true, a_pred = [], []
        b_true, b_pred = [], []
        running_loss = 0.0
        self.network.to(self.device)
        with torch.no_grad():
            for ((img1,img2), (d1,d2)) in tqdm(dataloader, total=len(dataloader), desc=desc):
                img1,img2 = img1.to(self.device), img2.to(self.device)
                target = d1 + d2
                target = target.to(self.device)
                dprob1 = self.network(img1)
                dprob2 = self.network(img2)
                output = conv_sum(dprob1,dprob2)
                running_loss += self.loss(torch.log(output + 1e-9), target.long()).item() * target.size(0)

                preds = output.argmax(dim=1)
                dpred1 = dprob1.argmax(dim=1)
                dpred2 = dprob2.argmax(dim=1)

                y_true.extend(target.cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

                a_true.extend(d1.cpu().tolist())
                a_pred.extend(dpred1.detach().cpu().tolist())

                b_true.extend(d2.cpu().tolist())
                b_pred.extend(dpred2.detach().cpu().tolist())
            

        test_loss = running_loss / len(dataloader.dataset)
        acc = accuracy_score(y_true, y_pred)
        d1acc = accuracy_score(a_true,a_pred)
        d2acc = accuracy_score(b_true,b_pred)

        if dataloader.dataset.dataset_name == "test":
            print(f"- {self.network.modelname} nesy model - Test loss: {test_loss} with total accuracy: {acc*100}% \n digit1 accuracy: {d1acc} and digit2 accuracy: {d2acc}")
        
        return {
            "loss": test_loss,
            "accuracy" : acc,
            "single-digit accuracy": (d1acc, d2acc)
        }
    


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
            print(f"Epoca {epoch+1}/{num_epoch} - {self.network.mnist_net.modelname} neural model - Train loss: {train_loss} with total accuracy: {running_train_accuracy*100}% \n digit1 accuracy: {d1acc} and digit2 accuracy: {d2acc}")
        
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
        

        print(f"- {self.network.mnist_net.modelname} neural model - Test loss: {test_loss} with total accuracy: {acc*100}% \n digit1 accuracy: {d1acc} and digit2 accuracy: {d2acc}")
        return {
            "loss": test_loss,
            "accuracy" : acc,
            "single-digit accuracy": (d1acc,d2acc)
        }
