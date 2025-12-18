import os
from abc import ABC, abstractmethod
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
import time
from sklearn.metrics import accuracy_score
import ltn

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
    
class MNISTNet_b0(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b0, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.modelname = "b0"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x):   
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        return x

class MNISTNet_b3(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b3, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.modelname = "b3"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x):   
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        return x
    
class MNISTSum2Net(nn.Module):
    def __init__(self, model):
        super(MNISTSum2Net, self).__init__()
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

class LogitsToPredicate(nn.Module):
    def __init__(self, mnist_model: MNISTNet):
        super(LogitsToPredicate, self).__init__()
        self.mnist_net = mnist_model

    def forward(self, x, d):
        d1 = self.mnist_net(x)
        p_d1 = F.softmax(d1, dim=1)
        p_d1 = torch.clamp(p_d1, min=1e-7, max=1.0 - 1e-7) # evito che con p più severi mi salti l'esecuzione 
        out = torch.gather(p_d1, 1, d)
        return out
    
    def classify(self, x: Tuple[torch.Tensor, torch.Tensor]):
        return self.mnist_net(x[0]), self.mnist_net(x[1])
    

def computePrediction(data, net):
    d1,d2 = data
    d1prob = F.softmax(net(d1), dim=1)
    d2prob = F.softmax(net(d2), dim=1)
    d1pred = d1prob.argmax(dim=1)
    d2pred = d2prob.argmax(dim=1)
    sum_pred = d1pred + d2pred
    return sum_pred, d1pred, d2pred


class Trainer_Sym():
    def __init__(self, train_loader, test_loader, model_dir, learning_rate, model : MNISTNet, device):
        self.model_dir = model_dir
        self.network = model
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss = nn.NLLLoss()
        self.device = device
        self.predicate = ltn.Predicate(LogitsToPredicate(self.network)).to(self.device)
        self.network.to(device)
        self.And = ltn.Connective(ltn.fuzzy_ops.AndProd())
        self.Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2), quantifier="e")
        self.Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        self.SatAgg = ltn.fuzzy_ops.SatAgg()

    def train(self, num_epoch):
        train_sats, train_losses, acc_train = [], [],[]
        d1_acc_train, d2_acc_train = [], []
        d_1 = ltn.Variable("d_1", torch.tensor(range(10)))
        d_2 = ltn.Variable("d_2", torch.tensor(range(10)))
        for epoch in range(num_epoch):
            if epoch in range(0, 4):
                p = 1
            if epoch in range(4, 8):
                p = 2
            if epoch in range(8, 12):
                p = 4
            if epoch in range(12, 20):
                p = 6
            y_true, y_pred = [], []
            a_true, a_pred = [], []
            b_true, b_pred = [], []
            total_inference_time = 0.0
            running_loss = 0.0
            train_sat = 0.0
            self.network.train()
            for ((img1,img2), (d1,d2)) in tqdm(self.train_loader, total=len(self.train_loader), desc='Train Loop'):
                self.optimizer.zero_grad()
                img1,img2 = img1.to(self.device), img2.to(self.device)
                target = d1+d2
                target = target.to(self.device)
                img_d1 = ltn.Variable("x", img1)
                img_d2 = ltn.Variable("y", img2)
                label = ltn.Variable("z", target)
                start_time = time.perf_counter()
                sat_agg = self.Forall(
                    ltn.diag(img_d1, img_d2, label),
                    self.Exists(
                        [d_1, d_2],
                        self.And(self.predicate(img_d1, d_1), self.predicate(img_d2, d_2)),
                        cond_vars=[d_1, d_2, label],
                        cond_fn=lambda d1, d2, z: torch.eq(d1.value + d2.value, z.value),
                        p=p
                    )
                ).value
                batch_time = time.perf_counter()-start_time
                total_inference_time += batch_time
                loss = 1. - sat_agg
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * target.size(0) 
                train_sat += sat_agg.item() * target.size(0)

                preds, d1pred, d2pred = computePrediction((img1,img2),self.network)
                
                y_true.extend(target.cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

                a_true.extend(d1.cpu().tolist())
                a_pred.extend(d1pred.detach().cpu().tolist())

                b_true.extend(d2.cpu().tolist())
                b_pred.extend(d2pred.detach().cpu().tolist())
            
            train_loss = running_loss / len(self.train_loader.dataset)
            train_losses.append(train_loss)
            running_train_accuracy = accuracy_score(y_true, y_pred)
            acc_train.append(running_train_accuracy)
            train_sat = train_sat / len(self.train_loader.dataset)
            train_sats.append(train_sat)
            d1acc = accuracy_score(a_true, a_pred)
            d2acc = accuracy_score(b_true, b_pred)
            d1_acc_train.append(d1acc)
            d2_acc_train.append(d2acc)
            avg_inference_time = total_inference_time / len(self.train_loader.dataset) * 1000
            
            #stats
            print(f"Epoca {epoch+1}/{num_epoch} - {self.network.modelname} nesy model - Train loss: {train_loss} with accuracy: {running_train_accuracy*100}% and Train Sat: {train_sat} in {avg_inference_time}ms\n digit1 accuracy: {d1acc*100} and digit2 accuracy: {d2acc*100}")
        
        torch.save(self.network.state_dict(), os.path.join(self.model_dir, f"{self.network.modelname}_nesy.pt"))
        return {
            "loss": train_losses,
            "accuracy" : acc_train,
            "single-digit accuracy": (max(d1_acc_train), max(d2_acc_train)),
            "sat" : max(train_sats)
        }
        

    def test(self):
        self.network.eval()
        y_true, y_pred = [], []
        a_true, a_pred = [], []
        b_true, b_pred = [], []
        running_loss = 0.0
        total_inference_time = 0.0
        test_sat = 0.0
        p=6
        d_1 = ltn.Variable("d_1", torch.tensor(range(10)))
        d_2 = ltn.Variable("d_2", torch.tensor(range(10)))
        with torch.no_grad():
            for ((img1,img2), (d1,d2)) in tqdm(self.test_loader, total=len(self.test_loader), desc='Test Loop'):
                img1,img2 = img1.to(self.device), img2.to(self.device)
                target = d1 + d2
                target = target.to(self.device)
                img_d1 = ltn.Variable("x", img1)
                img_d2 = ltn.Variable("y", img2)
                label = ltn.Variable("z", target)
                start_time = time.perf_counter()
                sat_agg = self.Forall(
                    ltn.diag(img_d1, img_d2, label),
                    self.Exists(
                        [d_1, d_2],
                        self.And(self.predicate(img_d1, d_1), self.predicate(img_d2, d_2)),
                        cond_vars=[d_1, d_2, label],
                        cond_fn=lambda d1, d2, z: torch.eq(d1.value + d2.value, z.value),
                        p=p
                    )
                ).value
                batch_time = time.perf_counter() - start_time
                total_inference_time += batch_time
                loss = 1. - sat_agg
                running_loss += loss.item() * target.size(0)
                test_sat += sat_agg.item() * target.size(0)
                preds, d1pred, d2pred = computePrediction((img1,img2),self.network)
                
                y_true.extend(target.cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

                a_true.extend(d1.cpu().tolist())
                a_pred.extend(d1pred.detach().cpu().tolist())

                b_true.extend(d2.cpu().tolist())
                b_pred.extend(d2pred.detach().cpu().tolist())

        test_loss = running_loss / len(self.test_loader.dataset)
        running_test_accuracy = accuracy_score(y_true, y_pred)
        test_sat = test_sat / len(self.test_loader.dataset)
        d1acc = accuracy_score(a_true, a_pred)
        d2acc = accuracy_score(b_true, b_pred)
        avg_inference_time = total_inference_time / len(self.test_loader.dataset) * 1000
        #aggiustare print
        print(f"- {self.network.modelname} nesy model - Test loss: {test_loss} with accuracy: {running_test_accuracy*100}% and Test Sat: {test_sat} in {avg_inference_time}ms\n digit1 accuracy: {d1acc*100} and digit2 accuracy: {d2acc*100}")
        return {
            "loss": test_loss,
            "accuracy" : running_test_accuracy,
            "single-digit accuracy": (d1acc, d2acc),
            "sat" : test_sat
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