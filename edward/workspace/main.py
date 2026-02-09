import time
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd
import numpy as np
import random
import os
import json
import utils

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def compGraph(losses, accuracy, modelname, data_dir):
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
        with open(os.path.join(data_dir, f"{filename}.json"), 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Training results salvati correttamente")
    except Exception as e:
        print(f"Errore durante il salvataggio: {e}")

def load_all_models(model_dir, models):
    w_files = [f for f in os.listdir(model_dir) if f.endswith('.h5') or f.endswith('.weights.h5')]
    for f in w_files:
        path = os.path.join(model_dir, f)
        filename = f.lower()
        is_sym = 0 if "neural" in filename else 1

        if "b0" in filename: key = "b0_nesy" if is_sym else "b0_neural"
        elif "b3" in filename: key = "b3_nesy" if is_sym else "b3_neural"
        elif "basic" in filename: key = "basic_nesy" if is_sym else "basic_neural"
        else: continue 
        
        input_ex = (tf.zeros((1, 28, 28, 1)), tf.zeros((1, 28, 28, 1)))
        if "b0" in key or "b3" in key: 
            input_ex = (tf.zeros((1, 64, 64, 3)), tf.zeros((1, 64, 64, 3)))
        _ = models[key](input_ex, training=False) #per creare i pesi a vuoto in modo da farli effettivamente caricare prima di ricevere dati
        models[key].load_weights(path)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20) 
parser.add_argument("--batch_size_train", type=int, default=32)
parser.add_argument("--batch_size_test", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--modeldir", type=str, default="./model/mnist_sum_2")
parser.add_argument("--datadir", type=str, default="./data")
parser.add_argument("--no_train", action="store_true")
args = parser.parse_args()
n_epochs = args.epochs
no_train = args.no_train
seed = args.seed
learning_rate = args.lr

# Setting Seeds
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

data_dir  = os.path.abspath(args.datadir)
model_dir = os.path.abspath(args.modeldir)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(os.path.join(data_dir,"basic"),exist_ok=True)
os.makedirs(os.path.join(data_dir,"b0"), exist_ok=True)
os.makedirs(os.path.join(data_dir,"b3"), exist_ok=True)

# DATASET

train_dataset = utils.MNISTSum2Dataset(data_dir, train=True)
test_dataset = utils.MNISTSum2Dataset(data_dir, train=False)

# DATALOADERS
train_loader_b, test_loader_b = utils.mnist_sum_2_loader(
    data_dir, args.batch_size_train, args.batch_size_test, "basic",
    train_dataset, test_dataset
)
train_loader_b0, test_loader_b0 = utils.mnist_sum_2_loader(
    data_dir, args.batch_size_train, args.batch_size_test, "mnistb0",
    train_dataset, test_dataset
)
train_loader_b3, test_loader_b3 = utils.mnist_sum_2_loader(
    data_dir, args.batch_size_train, args.batch_size_test, "mnistb3",
    train_dataset, test_dataset
)

# Classificatori
basic_sym = utils.MNISTNet_basic()
basic_nsym = utils.MNISTNet_basic()
b0_sym = utils.MNISTNet_Efficient("b0")
b0_nsym = utils.MNISTNet_Efficient("b0")
b3_sym = utils.MNISTNet_Efficient("b3")
b3_nsym = utils.MNISTNet_Efficient("b3")

# Modelli per la somma
b_ns = utils.MNISTSum2Net(basic_nsym)
b_s = utils.MNISTSum2Net_nesy(basic_sym)
b0_ns = utils.MNISTSum2Net(b0_nsym)
b0_s = utils.MNISTSum2Net_nesy(b0_sym)
b3_ns = utils.MNISTSum2Net(b3_nsym)
b3_s = utils.MNISTSum2Net_nesy(b3_sym)

models_dict = {
    "basic_neural": b_ns, "basic_nesy": b_s, 
    "b0_neural": b0_ns, "b0_nesy": b0_s, 
    "b3_neural": b3_ns, "b3_nesy": b3_s
}

# TRAINERS
trainer_NoSym_basic = utils.Trainer_NoSym(train_loader_b, test_loader_b, model_dir, learning_rate, b_ns)
trainer_NoSym_b0 = utils.Trainer_NoSym(train_loader_b0, test_loader_b0, model_dir, learning_rate, b0_ns)
trainer_NoSym_b3 = utils.Trainer_NoSym(train_loader_b3, test_loader_b3, model_dir, learning_rate, b3_ns)

trainer_Sym_basic = utils.Trainer_Sym(train_loader_b, test_loader_b, model_dir,b_s, learning_rate)
trainer_Sym_b0 = utils.Trainer_Sym(train_loader_b0, test_loader_b0, model_dir,b0_s,learning_rate)
trainer_Sym_b3 = utils.Trainer_Sym(train_loader_b3, test_loader_b3, model_dir, b3_s, learning_rate)

if no_train:
    load_all_models(model_dir, models_dict)
    with open(os.path.join(data_dir, "training_results.json"), "r", encoding="utf-8") as f:
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
            "train_nesy": rb_train_s, 
            "train_neural": rb_train_ns 
        },
        "EfficientNet-B0": {
            "train_nesy": rb0_train_s, 
            "train_neural": rb0_train_ns 
        },
        "EfficientNet-B3": { 
            "train_nesy": rb3_train_s, 
            "train_neural": rb3_train_ns 
        },
    }   
    saveJsonResults("training_results", training_results)

# TESTING
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

# generazione risultati
rows = list(results.keys())
cols = ["Train (NeSy)", "Test (NeSy)","Train (Neural)", "Test (Neural)"]

def fmt_cell(loss, acc, single_acc):
    d1, d2 = single_acc
    return f"L:{loss:.4f}\nA:{acc*100:.1f}%\nD:{d1*100:.1f}/{d2*100:.1f}%"

df = pd.DataFrame(index=rows, columns=cols)
for model in rows:
    r = results[model]
    if "basic" in model: modelname = "basic"
    elif "B0" in model: modelname = "b0"
    else: modelname = "b3"

    compGraph({"nesy": r["train_nesy"]["loss"], "neural": r["train_neural"]["loss"]},
            {"nesy": r["train_nesy"]["accuracy"], "neural": r["train_neural"]["accuracy"]},
            modelname, data_dir)
    
    tr_nesy_loss = min(r["train_nesy"]["loss"]) if r["train_nesy"]["loss"] else 0
    tr_nesy_acc = max(r["train_nesy"]["accuracy"]) if r["train_nesy"]["accuracy"] else 0
    tr_neur_loss = min(r["train_neural"]["loss"]) if r["train_neural"]["loss"] else 0
    tr_neur_acc = max(r["train_neural"]["accuracy"]) if r["train_neural"]["accuracy"] else 0

    df.loc[model, "Train (NeSy)"] = fmt_cell(tr_nesy_loss, tr_nesy_acc, r["train_nesy"]["single-digit accuracy"])
    df.loc[model, "Test (NeSy)"] = fmt_cell(r["test_nesy"]["loss"], r["test_nesy"]["accuracy"], r["test_nesy"]["single-digit accuracy"])
    df.loc[model, "Train (Neural)"] = fmt_cell(tr_neur_loss, tr_neur_acc, r["train_neural"]["single-digit accuracy"])
    df.loc[model, "Test (Neural)"] = fmt_cell(r["test_neural"]["loss"], r["test_neural"]["accuracy"], r["test_neural"]["single-digit accuracy"])

def save_table_image(df : pd.DataFrame, out_png, figsize=(12, 4.5), dpi=200, header_color="#f0f0f0"):
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

save_table_image(df, out_png=os.path.join(data_dir, "results.png"))