import torch
import argparse
import matplotlib.pyplot as plt #
from matplotlib.table import Table
import pandas as pd
import numpy as np, random, torch
import os
from datetime import datetime
import pandas as pd
import utils 
import json

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
        is_sym = 0 if "nosym" in filename else 1

        if "b0" in filename: key = "b0_sym" if is_sym else "b0_nosym"
        elif "b3" in filename: key = "b3_sym" if is_sym else "b3_nosym"
        elif "basic" in filename: key = "basic_sym" if is_sym else "basic_nosym"
        else: raise KeyError(f"Nessun modello per il file '{f}' (chiave inferita: {key}).")
        
        state = torch.load(path, map_location=device)
        models[key].load_state_dict(state)

parser = argparse.ArgumentParser()

# Parameters
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size_train", type=int, default=32)
parser.add_argument("--batch_size_test", type=int, default=32)
parser.add_argument("--lr", type=int, default=1e-3)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--modeldir", type=str, default="./model/mnist_sum_2")
parser.add_argument("--datadir", type=str, default="./data")
parser.add_argument("--no_train", action="store_true")
args = parser.parse_args()
n_epochs = args.epochs
no_train = args.no_train
seed = args.seed
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_train
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


#data loader in cui passo il dataset che a cui verrà modificata la transform

train_loader_b, test_loader_b = utils.mnist_sum_2_loader(
    data_dir, batch_size_train, batch_size_test, "basic",
    train_dataset=train_dataset, test_dataset=test_dataset
)
train_loader_b0, test_loader_b0 = utils.mnist_sum_2_loader(
    data_dir, batch_size_train, batch_size_test, "mnistb0",
    train_dataset=train_dataset, test_dataset=test_dataset
)
train_loader_b3, test_loader_b3 = utils.mnist_sum_2_loader(
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
b_s = utils.MNISTSum2Net(basic_sym)
b0_s = utils.MNISTSum2Net(b0_sym)
b3_s = utils.MNISTSum2Net(b3_sym)

models = {"basic_nosym":b_ns, "basic_sym":b_s, 
        "b0_nosym":b0_ns, "b0_sym":b0_s, 
        "b3_nosym":b3_ns, "b3_sym":b3_s}

#trainer no sym
trainer_NoSym_basic = utils.Trainer_NoSym(train_loader_b, test_loader_b, model_dir,learning_rate, b_ns, device)
trainer_NoSym_b0 = utils.Trainer_NoSym(train_loader_b0, test_loader_b0, model_dir,learning_rate, b0_ns , device)
trainer_NoSym_b3 = utils.Trainer_NoSym(train_loader_b3, test_loader_b3, model_dir,learning_rate, b3_ns , device)

#trainer sym
trainer_Sym_basic = utils.Trainer_Sym(train_loader_b, test_loader_b, model_dir,learning_rate, b_s, device)
trainer_Sym_b0 = utils.Trainer_Sym(train_loader_b0, test_loader_b0, model_dir,learning_rate, b0_s, device)
trainer_Sym_b3 = utils.Trainer_Sym(train_loader_b3, test_loader_b3, model_dir,learning_rate, b3_s, device)

#training

if no_train is True:
    load_all_models(model_dir, models)
    with open(data_dir+"/training_results.json", "r", encoding="utf-8") as f:
        training_results = json.load(f)

else:
    print("Inizio training dei modelli")
    rb_train_s = trainer_Sym_basic.train(n_epochs)
    rb_train_ns = trainer_NoSym_basic.train(n_epochs)
    rb0_train_ns = trainer_NoSym_b0.train(n_epochs)
    rb0_train_s = trainer_Sym_b0.train(n_epochs)
    rb3_train_ns = trainer_NoSym_b3.train(n_epochs)
    rb3_train_s = trainer_Sym_b3.train(n_epochs)
    training_results = {
        "MNISTNet_basic": {
            "train_sym":  rb_train_s,
            "train_nosym": rb_train_ns,
        },
        "EfficientNet-B0": {
            "train_sym":  rb0_train_s,
            "train_nosym": rb0_train_ns,
        },
        "EfficientNet-B3": {
            "train_sym":  rb3_train_s,
            "train_nosym": rb3_train_ns,
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
        "train_sym":  training_results["MNISTNet_basic"]["train_sym"],
        "test_sym":   rb_test_s,
        "train_nosym": training_results["MNISTNet_basic"]["train_nosym"],
        "test_nosym": rb_test_ns,
    },
    "EfficientNet-B0": {
        "train_sym":  training_results["EfficientNet-B0"]["train_sym"],
        "test_sym":   rb0_test_s,
        "train_nosym": training_results["EfficientNet-B0"]["train_nosym"],
        "test_nosym": rb0_test_ns,
    },
    "EfficientNet-B3": {
        "train_sym":  training_results["EfficientNet-B3"]["train_sym"],
        "test_sym":   rb3_test_s,
        "train_nosym": training_results["EfficientNet-B3"]["train_nosym"],
        "test_nosym": rb3_test_ns,
    },
}

# salvataggio risultati

rows = list(results.keys())
cols = ["Train (Sym)", "Test (Sym)","Train (NoSym)", "Test (NoSym)"]

def fmt_cell(loss, acc):
    return f"{loss:.4f} / {acc*100:.2f}%"

df = pd.DataFrame(index=rows, columns=cols)
for model in rows:
    r = results[model]
    df.loc[model, "Train (Sym)"] = fmt_cell(r["train_sym"]["loss"], r["train_sym"]["accuracy"])
    df.loc[model, "Test (Sym)"] = fmt_cell(r["test_sym"]["loss"], r["test_sym"]["accuracy"])
    df.loc[model, "Train (NoSym)"] = fmt_cell(r["train_nosym"]["loss"], r["train_nosym"]["accuracy"])
    df.loc[model, "Test (NoSym)"] = fmt_cell(r["test_nosym"]["loss"], r["test_nosym"]["accuracy"])

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