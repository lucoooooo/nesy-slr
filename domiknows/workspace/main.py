import torch
import matplotlib.pyplot as plt #
from matplotlib.table import Table
import pandas as pd
import numpy as np, random, torch
import argparse
import os
import json
from datetime import datetime
import pandas as pd
import utils 
import time
def saveJsonResults(filename, data):
        try:
            with open(data_dir+f"/{filename}.json", 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Training results salvati correttamente")
        except Exception as e:
            print(f"Errore durante il salvataggio: {e}")


def load_all_models(model_dir, models, device):
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    #device="cpu"
    for f in pt_files:
        path = model_dir + "/" + f
        state = torch.load(path, map_location=device)
        filename = f.lower()
        is_sym = 0 if "neural" in filename else 1

        if "b0" in filename: key = "b0_nesy" if is_sym else "b0_neural"
        elif "b3" in filename: key = "b3_nesy" if is_sym else "b3_neural"
        elif "basic" in filename: key = "basic_nesy" if is_sym else "basic_neural"
        else: raise KeyError(f"Nessun modello per il file '{f}' (chiave inferita: {key}).")

        models[key].load_state_dict(state)


parser = argparse.ArgumentParser()

# Parameters
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size_train", type=int, default=32)
parser.add_argument("--batch_size_test", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--modeldir", type=str, default="./model/mnist_sum_2")
parser.add_argument("--datadir", type=str, default="./data")
parser.add_argument("--no_train", action="store_true", default = False)

args = parser.parse_args()
n_epochs = args.epochs
no_train = args.no_train
seed = args.seed
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
learning_rate = args.lr


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir  = os.path.abspath(args.datadir)
model_dir = os.path.abspath(args.modeldir)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
#DATASET uguale per tutti

train_dataset = utils.MNISTSum2Dataset(
    root=data_dir,
    train=True,
    download=True,
    transform=utils.mnistbasic_img_transform 
)
test_dataset = utils.MNISTSum2Dataset(
    root=data_dir,
    train=False,
    download=True,
    transform=utils.mnistbasic_img_transform
)


train_loader_b, test_loader_b = utils.mnist_sum_2_loader(
    data_dir, batch_size_train, batch_size_test, "basic",
    train_dataset=train_dataset, test_dataset=test_dataset,
)
train_loader_b0, test_loader_b0= utils.mnist_sum_2_loader(
    data_dir, batch_size_train, batch_size_test, "mnistb0",
    train_dataset=train_dataset, test_dataset=test_dataset,
)
train_loader_b3, test_loader_b3 = utils.mnist_sum_2_loader(
    data_dir, batch_size_train, batch_size_test, "mnistb3",
    train_dataset=train_dataset, test_dataset=test_dataset,
)

#modelli

basic_nsym = utils.MNISTNet_basic()
b0_nsym = utils.MNISTNet_b0()
b3_nsym = utils.MNISTNet_b3()


b_ns = utils.MNISTSum2Net_NoSym(basic_nsym)
b0_ns = utils.MNISTSum2Net_NoSym(b0_nsym)
b3_ns = utils.MNISTSum2Net_NoSym(b3_nsym)


models = {"basic_neural":b_ns,  
        "b0_neural":b0_ns,
        "b3_neural":b3_ns}


#trainer no sym
trainer_NoSym_basic = utils.Trainer_NoSym(train_loader_b, test_loader_b,model_dir, learning_rate, b_ns, device)
trainer_NoSym_b0 = utils.Trainer_NoSym(train_loader_b0, test_loader_b0, model_dir,learning_rate, b0_ns , device)
trainer_NoSym_b3 = utils.Trainer_NoSym(train_loader_b3, test_loader_b3, model_dir,learning_rate, b3_ns , device)

if no_train is True:
    load_all_models(model_dir, models, device)
    with open(data_dir+"/training_results.json", "r", encoding="utf-8") as f:
        training_results = json.load(f)

else:
    
    t0_b_ns = time.perf_counter() 
    rb_train_ns = trainer_NoSym_basic.train(n_epochs)
    print(f"Training modello basic neural terminato in: {utils.time_delta_now(t0_b_ns)}")

    t0_b0_ns = time.perf_counter() 
    rb0_train_ns = trainer_NoSym_b0.train(n_epochs)
    print(f"Training modello b0 neural terminato in: {utils.time_delta_now(t0_b0_ns)}")

    t0_b3_ns = time.perf_counter() 
    rb3_train_ns = trainer_NoSym_b3.train(n_epochs)
    print(f"Training modello b3 neural terminato in: {utils.time_delta_now(t0_b3_ns)}")

    training_results = {
        "MNISTNet_basic": {
            "train_neural": rb_train_ns,
        },
        "EfficientNet-B0": {
            "train_neural": rb0_train_ns,
        },
        "EfficientNet-B3": {
            "train_neural": rb3_train_ns,
        },
    }   
    saveJsonResults("training_results", training_results)


#testing
print("Inizio testing dei modelli")
rb_test_ns = trainer_NoSym_basic.test()
rb0_test_ns = trainer_NoSym_b0.test()
rb3_test_ns = trainer_NoSym_b3.test()


testing_results = {
        "MNISTNet_basic": {
            "test_neural": rb_test_ns,
        },
        "EfficientNet-B0": {
            "test_neural": rb0_test_ns,
        },
        "EfficientNet-B3": {
            "test_neural": rb3_test_ns,
        },
}
saveJsonResults("testing_results", testing_results)
