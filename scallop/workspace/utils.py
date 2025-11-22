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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import scallopy

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

class MNISTSum2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        # Contains a MNIST dataset
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

    
    def clone_with_transform(base_ds: MNISTSum2Dataset, train_flag: bool, transform):
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
    
class MNISTNet_b0(nn.Module):
    def __init__(self, N=10):
        super(MNISTNet_b0, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.modelname = "b0"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x):   
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        return F.softmax(x,dim=1)

class MNISTNet_b3(nn.Module):
    def __init__(self, N=10):
        super(MNISTNet_b3, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.modelname = "b3"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x):   
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        return F.softmax(x,dim=1)

class MNISTSum2Net_Sym(nn.Module):
    def __init__(self, model, provenance, k):
        super(MNISTSum2Net_Sym, self).__init__()

        self.mnist_net = model

        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
        self.scl_ctx.add_rule("sum_2(a + b) :- digit_1(a), digit_2(b)")

        self.sum_2 = self.scl_ctx.forward_function("sum_2", output_mapping=[(i,) for i in range(19)])

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        (a_imgs, b_imgs) = x

        a_distrs = self.mnist_net(a_imgs)
        b_distrs = self.mnist_net(b_imgs) 

        return self.sum_2(digit_1=a_distrs, digit_2=b_distrs) 

class Trainer_Sym():
    def __init__(self, train_loader, test_loader, model_dir, learning_rate, model : MNISTSum2Net_Sym, device):
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
                output = self.network((d1,d2))
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
        torch.save(self.network.state_dict(), os.path.join(self.model_dir, f"{self.network.mnist_net.modelname}_sym.pt"))
        return {
            "loss": min(train_losses),
            "accuracy" : max(acc_train)
        }
        

    def test(self):
        self.network.eval()
        y_true, y_pred = [], []
        running_loss = 0.0
        with torch.no_grad():
            for ((d1,d2), target) in tqdm(self.test_loader, total=len(self.test_loader), desc='Test Loop'):
                d1,d2 = d1.to(self.device), d2.to(self.device)
                target = target.to(self.device)
                output = self.network((d1,d2))
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
    
class MNISTSum2Net_NoSym(nn.Module):
    def __init__(self, model):
        super(MNISTSum2Net_NoSym, self).__init__()
        self.mnist_net = model


    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        (a_imgs, b_imgs) = x


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
    def __init__(self, train_loader, test_loader, model_dir, learning_rate, model : MNISTSum2Net_NoSym, device):
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
                output = conv_sum(dprob1, dprob2)
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
                output = conv_sum(dprob1, dprob2)
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