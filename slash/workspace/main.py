import torch
import argparse
import matplotlib.pyplot as plt #
from matplotlib.table import Table
import pandas as pd
import numpy as np, random, torch
import sys
import re
import os
from datetime import datetime
import pandas as pd
import utils 
import json
import time

def compGraph(accuracy, modelname, data_dir, losses = None):
    if losses is not None:
        plt.figure(figsize=(10,6))
        plt.plot(losses["nesy"], label='Neuro-Symbolic', color='blue', linewidth=2)
        plt.plot(losses["neural"], label='Neural baseline (Black Box)', color='red',linestyle='--', linewidth=2)
        plt.title("Confronto Loss in training: NeSy vs Neural Baseline", fontsize=14)
        plt.xlabel("Epoche", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.ylim(0, 4.0)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=12)
        plt.savefig(os.path.join(data_dir, f"{modelname}/loss_{modelname}_result.png"), dpi=300)
        plt.show()
        plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(accuracy["nesy"], label='Neuro-Symbolic', color='blue', linewidth=2)
    plt.plot(accuracy["neural"], label='Neural baseline (Black Box)', color='red',linestyle='--', linewidth=2)
    plt.title("Confronto Accuracy in training: NeSy vs Neural Baseline", fontsize=14)
    plt.xlabel("Epoche", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(data_dir, f"{modelname}/accuracy_{modelname}_result.png"), dpi=300)
    plt.show()
    plt.close()

def saveJsonResults(filename, data):
        try:
            with open(data_dir+f"/{filename}.json", 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Training results salvati correttamente")
        except Exception as e:
            print(f"Errore durante il salvataggio: {e}")

def load_all_models(model_dir, models):
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') or f.endswith('.pth')]
    for f in pt_files:
        path = model_dir + "/" + f
        filename = f.lower()
        is_sym = 0 if "neural" in filename else 1

        if "b0" in filename: key = "b0_nesy" if is_sym else "b0_neural"
        elif "b3" in filename: key = "b3_nesy" if is_sym else "b3_neural"
        elif "basic" in filename: key = "basic_nesy" if is_sym else "basic_neural"
        else: raise KeyError(f"Nessun modello per il file '{f}' (chiave inferita: {key}).")
        
        state = torch.load(path, map_location=device)
        models[key].load_state_dict(state)



parser = argparse.ArgumentParser()

# Parameters
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size-train", type=int, default=32)
parser.add_argument("--batch-size-test", type=int, default=32)
parser.add_argument("--p-num", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--k", type=int, default=0)
parser.add_argument("--method",choices=["exact","top_k","same"], default = "same")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--modeldir", type=str, default="./model/mnist_sum_2")
parser.add_argument("--datadir", type=str, default="./data")
parser.add_argument("--no-train", action="store_true")
args = parser.parse_args()
n_epochs = args.epochs
no_train = args.no_train
seed = args.seed
k = args.k
p_num = args.p_num
method = args.method
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_train
learning_rate = args.lr
print(method)

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

#DATASET uguale per tutti

train_dataset = utils.MNISTSum2Dataset(
    root=data_dir,
    train=True,
    download=True,
    transform=utils.mnistbasic_img_transform,
    seed = seed
)
test_dataset = utils.MNISTSum2Dataset(
    root=data_dir,
    train=False,
    download=True,
    transform=utils.mnistbasic_img_transform,
    seed=seed
)


#data loader in cui passo il dataset che a cui verrà modificata la transform

train_loader_b_ns, test_loader_b_ns, train_loader_b_s= utils.mnist_sum_2_loader(
    data_dir, batch_size_train, batch_size_test, "basic",
    train_dataset=train_dataset, test_dataset=test_dataset
)
train_loader_b0_ns, test_loader_b0_ns, train_loader_b0_s= utils.mnist_sum_2_loader(
    data_dir, batch_size_train, batch_size_test, "mnistb0",
    train_dataset=train_dataset, test_dataset=test_dataset
)
train_loader_b3_ns, test_loader_b3_ns, train_loader_b3_s= utils.mnist_sum_2_loader(
    data_dir, batch_size_train, batch_size_test, "mnistb3",
    train_dataset=train_dataset, test_dataset=test_dataset
)

#modelli
basic_sym = utils.MNISTNet_basic()
basic_nsym = utils.MNISTNet_basic()
b0_sym = utils.MNISTNet_b0()
b0_nsym = utils.MNISTNet_b0()
b3_sym = utils.MNISTNet_b3()
b3_nsym = utils.MNISTNet_b3()

b_ns = utils.MNISTSum2Net(basic_nsym)
b0_ns = utils.MNISTSum2Net(b0_nsym)
b3_ns = utils.MNISTSum2Net(b3_nsym) 

models = {"basic_neural":b_ns, "basic_nesy":basic_sym, 
        "b0_neural":b0_ns, "b0_nesy":b0_sym, 
        "b3_neural":b3_ns, "b3_nesy":b3_sym}

#trainer no sym
trainer_NoSym_basic = utils.Trainer_NoSym(train_loader_b_ns, test_loader_b_ns, model_dir,learning_rate, b_ns, device)
trainer_NoSym_b0 = utils.Trainer_NoSym(train_loader_b0_ns, test_loader_b0_ns, model_dir,learning_rate, b0_ns , device)
trainer_NoSym_b3 = utils.Trainer_NoSym(train_loader_b3_ns, test_loader_b3_ns, model_dir,learning_rate, b3_ns , device)

#trainer sym
trainer_Sym_basic = utils.Trainer_Sym(train_loader_b_s, train_loader_b_ns, test_loader_b_ns, model_dir,learning_rate, basic_sym, method, p_num, k,device)
trainer_Sym_b0 = utils.Trainer_Sym(train_loader_b0_s, train_loader_b0_ns, test_loader_b0_ns, model_dir,learning_rate, b0_sym, method, p_num, k,device)
trainer_Sym_b3 = utils.Trainer_Sym(train_loader_b3_s, train_loader_b3_ns, test_loader_b3_ns, model_dir,learning_rate, b3_sym, method, p_num, k,device)

#training
if no_train is True:
    load_all_models(model_dir, models)
    with open(data_dir+"/training_results.json", "r", encoding="utf-8") as f:
        training_results = json.load(f)
else:
    print("Inizio training dei modelli")
    
    t0_b_s = time.perf_counter() 
    rb_train_s = trainer_Sym_basic.train(n_epochs)
    print(f"Training modello basic nesy terminato in: {utils.time_delta_now(t0_b_s)}")

    t0_b_ns = time.perf_counter() 
    rb_train_ns = trainer_NoSym_basic.train(n_epochs)
    print(f"Training modello basic neural terminato in: {utils.time_delta_now(t0_b_ns)}")

    t0_b0_ns = time.perf_counter() 
    rb0_train_ns = trainer_NoSym_b0.train(n_epochs)
    print(f"Training modello b0 neural terminato in: {utils.time_delta_now(t0_b0_ns)}")

    t0_b0_s = time.perf_counter() 
    rb0_train_s = trainer_Sym_b0.train(n_epochs)
    print(f"Training modello b0 nesy terminato in: {utils.time_delta_now(t0_b0_s)}")

    t0_b3_ns = time.perf_counter() 
    rb3_train_ns = trainer_NoSym_b3.train(n_epochs)
    print(f"Training modello b3 neural terminato in: {utils.time_delta_now(t0_b3_ns)}")

    t0_b3_s = time.perf_counter() 
    rb3_train_s = trainer_Sym_b3.train(n_epochs)
    print(f"Training modello b3 nesy terminato in: {utils.time_delta_now(t0_b3_s)}")
    
    training_results = {
        "MNISTNet_basic": {
            "train_nesy":  rb_train_s,
            "train_neural": rb_train_ns,
        },
        "EfficientNet-B0": {
            "train_nesy":  rb0_train_s,
            "train_neural": rb0_train_ns,
        },
        "EfficientNet-B3": {
            "train_nesy":  rb3_train_s,
            "train_neural": rb3_train_ns,
        },
    }   
    saveJsonResults("training_results", training_results)


#testing
print("Inizio testing dei modelli")
rb_test_ns = trainer_NoSym_basic.test()
rb_test_s = trainer_Sym_basic.test()
rb0_test_ns = trainer_NoSym_b0.test()
rb0_test_s = trainer_Sym_b0.test()
rb3_test_ns = trainer_NoSym_b3.test()
rb3_test_s = trainer_Sym_b3.test()


results = {
    "MNISTNet_basic": {
        "train_nesy":  training_results["MNISTNet_basic"]["train_nesy"],
        "test_nesy":   rb_test_s,
        "train_neural": training_results["MNISTNet_basic"]["train_neural"],
        "test_neural": rb_test_ns,
    },
    "EfficientNet-B0": {
        "train_nesy":  training_results["EfficientNet-B0"]["train_nesy"],
        "test_nesy":   rb0_test_s,
        "train_neural": training_results["EfficientNet-B0"]["train_neural"],
        "test_neural": rb0_test_ns,
    },
    "EfficientNet-B3": {
        "train_nesy":  training_results["EfficientNet-B3"]["train_nesy"],
        "test_nesy":   rb3_test_s,
        "train_neural": training_results["EfficientNet-B3"]["train_neural"],
        "test_neural": rb3_test_ns,
    },
}

# salvataggio risultati

rows = list(results.keys())

cols = ["Train (NeSy)", "Test (NeSy)","Train (Neural)", "Test (Neural)"]

def fmt_cell(loss, acc, single_acc):
    d1, d2 = single_acc
    return f"L:{loss:.4f}\nA:{acc*100:.1f}%\nD:{d1*100:.1f}/{d2*100:.1f}%"

df = pd.DataFrame(index=rows, columns=cols)
for model in rows:
    r = results[model]

    if "basic" in model:
        modelname = "basic"
    elif "B0" in model:
        modelname = "b0"
    else:
        modelname = "b3"
        
    compGraph(
            {"nesy": r["train_nesy"]["accuracy"], "neural": r["train_neural"]["accuracy"]},
            modelname,
            data_dir
            )
    
    df.loc[model, "Train (NeSy)"] = fmt_cell(min(r["train_nesy"]["loss"]), max(r["train_nesy"]["accuracy"]), r["train_nesy"]["single-digit accuracy"])
    df.loc[model, "Test (NeSy)"] = fmt_cell(r["test_nesy"]["loss"], r["test_nesy"]["accuracy"], r["test_nesy"]["single-digit accuracy"])
    df.loc[model, "Train (Neural)"] = fmt_cell(min(r["train_neural"]["loss"]), max(r["train_neural"]["accuracy"]), r["train_neural"]["single-digit accuracy"])
    df.loc[model, "Test (Neural)"] = fmt_cell(r["test_neural"]["loss"], r["test_neural"]["accuracy"], r["test_neural"]["single-digit accuracy"])

def save_table_image(df : pd.DataFrame, out_png="./data/results.png", figsize=(12, 4.5), dpi=200, header_color="#f0f0f0"):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('off')

    tbl = Table(ax, bbox=[0,0,1,1])

    n_rows, n_cols = df.shape
    cell_w = 1.0 / (n_cols + 1)
    cell_h = 1.0 / (n_rows + 1)

    tbl.add_cell(0,0, cell_w, cell_h, text="", loc="center", facecolor=header_color)
    for j, col in enumerate(df.columns, start=1):
        tbl.add_cell(0,j,cell_w,cell_h, text=str(col), loc="center", facecolor=header_color)

    for i, row in enumerate(df.index, start=1):
        tbl.add_cell(i, 0, cell_w, cell_h, text=str(row), loc="left", facecolor=header_color)
        for j, col in enumerate(df.columns, start=1):
            tbl.add_cell(i,j,cell_w,cell_h, text=str(df.loc[row,col]), loc="center", facecolor="white")

    ax.add_table(tbl)

    plt.savefig(out_png, bbox_inches="tight", pad_inches = 0.5)
    plt.close(fig)

save_table_image(df)