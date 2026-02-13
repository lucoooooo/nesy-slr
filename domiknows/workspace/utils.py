
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
import time


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
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229,0.224,0.225]),
])

mnistb3_img_transform = mnistb0_img_transform

def time_delta_now(t_start: float, simple_format=True) -> str:
    a = t_start
    b = time.perf_counter() 
    c = b - a  
    days = int(c // 86400)
    hours = int(c // 3600 % 24)
    minutes = int(c // 60 % 60)
    seconds = int(c % 60)
    millisecs = round(c % 1 * 1000)
    if simple_format:
        return f"{hours}h:{minutes}m:{seconds}s"

    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds", c


class MNISTSum2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.mnist_dataset = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        )
        self.index_map = list(range(len(self.mnist_dataset)))
        random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.index_map) / 2)

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


def mnist_sum_2_loader(
    data_dir,
    batch_size_train,
    batch_size_test,
    modeltype: str,
    train_dataset: MNISTSum2Dataset,
    test_dataset: MNISTSum2Dataset
):
    if modeltype == "mnistb0":
        mnist_img_transform = mnistb0_img_transform
    elif modeltype == "mnistb3":
        mnist_img_transform = mnistb3_img_transform
    else:
        mnist_img_transform = mnistbasic_img_transform

    
    def clone_with_transform(base_ds: "MNISTSum2Dataset", train_flag: bool, transform):
        ds = MNISTSum2Dataset(
            data_dir, train=train_flag, download=True, transform=transform
        )
        ds.index_map = list(base_ds.index_map)
        return ds

    train_ds = clone_with_transform(train_dataset, True, mnist_img_transform)
    
    test_ds = clone_with_transform(test_dataset, False, mnist_img_transform)
    
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

class MNISTNet_basic(nn.Module):
    def __init__(self, num_classes=10):
        super(MNISTNet_basic, self).__init__()
        self.modelname = "basic"
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
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
        return x
    
class MNISTNet_b0(nn.Module):
    def __init__(self, N=10):
        super(MNISTNet_b0, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.modelname = "b0"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x):   
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        return x

class MNISTNet_b3(nn.Module):
    def __init__(self, N=10):
        super(MNISTNet_b3, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.modelname = "b3"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x):   
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        return x

def conv_sum(d1, d2):  
    B = d1.size(0)
    x = d1.unsqueeze(0)                     
    w = d2.unsqueeze(1).flip(2)             
    y = nn.functional.conv1d(x, w, groups=B, padding=9)  
    p_sum = y.squeeze(0)[:, :19]             
    p_sum = p_sum / (p_sum.sum(dim=1, keepdim=True) + 1e-12)
    return p_sum
    

class MNISTSum2Net_NoSym(nn.Module):
    def __init__(self, model):
        super(MNISTSum2Net_NoSym, self).__init__()
        self.mnist_net = model
        self.feature_dim = 10

        combined_dim = self.feature_dim * 2
        self.sum_classifier = nn.Linear(combined_dim, 19)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        (a_imgs, b_imgs) = x

        a_feat = self.mnist_net(a_imgs)
        b_feat = self.mnist_net(b_imgs)
        
        a_pred = (F.softmax(a_feat, dim=1)).argmax(dim=1)
        b_pred = (F.softmax(b_feat, dim=1)).argmax(dim=1)

        combined_feature = torch.cat((a_feat,b_feat), dim=1)

        sumpred = F.softmax(self.sum_classifier(combined_feature), dim=1)

        return sumpred, a_pred, b_pred


class Trainer_NoSym:
    def __init__(self, train_loader, test_loader, model_dir,learning_rate, model : MNISTSum2Net_NoSym, device):
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