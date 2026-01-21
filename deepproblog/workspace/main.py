import torch
import matplotlib.pyplot as plt #
from matplotlib.table import Table
import pandas as pd
import numpy as np, random
import os
import json
import pandas as pd
import utils 
from deepproblog.evaluate import get_confusion_matrix
import argparse
import torchvision
import time

def compGraph(losses, modelname, data_dir,accuracy=None):
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

    if accuracy is not None:
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

def load_all_models(model_dir, models, device):
    pt_files = [f for f in os.listdir(model_dir) if f.endswith('.pt') or f.endswith('.pth')]
    for f in pt_files:
        path = model_dir + "/" + f
        filename = f.lower()
        is_sym = 0 if "neural" in filename else 1

        if "b0" in filename: key = "b0_nesy" if is_sym else "b0_neural"
        elif "b3" in filename: key = "b3_nesy" if is_sym else "b3_neural"
        elif "basic" in filename: key = "basic_nesy" if is_sym else "basic_neural"
        else: raise KeyError(f"Nessun modello per il file '{f}' (chiave inferita: {key}).")
        
        if is_sym:
            models[key].load_state_dict(path, device)
        else:
            state = torch.load(path, map_location=device)
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
parser.add_argument("--no_train", action="store_true")
parser.add_argument("--method", choices=["exact","approximate"], type=str, default="approximate")
args = parser.parse_args()
n_epochs = args.epochs
no_train = args.no_train
seed = args.seed
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_train
learning_rate = args.lr
method = args.method

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
os.makedirs(os.path.join(data_dir,"basic"),exist_ok=True)
os.makedirs(os.path.join(data_dir,"b0"), exist_ok=True)
os.makedirs(os.path.join(data_dir,"b3"), exist_ok=True)

#DATASET uguale per tutti

train_dataset_nosym = utils.MNISTSum2Dataset(
    root=data_dir,
    train=True,
    download=True,
    transform=utils.mnistbasic_img_transform,
    seed = seed
)
test_dataset_nosym= utils.MNISTSum2Dataset(
    root=data_dir,
    train=False,
    download=True,
    transform=utils.mnistbasic_img_transform,
    seed = seed
)


dataset_sym = utils.createSYMdataset(train_dataset_nosym, test_dataset_nosym)


test_loader_singledigit_basic = torch.utils.data.DataLoader(
        dataset_sym["basic"]["testset"].mnist_dataset,
        batch_size=batch_size_test,
        shuffle=False
)
test_loader_singledigit_b0 = torch.utils.data.DataLoader(
        dataset_sym["b0"]["testset"].mnist_dataset,
        batch_size=batch_size_test,
        shuffle=False
)
test_loader_singledigit_b3 = torch.utils.data.DataLoader(
        dataset_sym["b3"]["testset"].mnist_dataset,
        batch_size=batch_size_test,
        shuffle=False
)
#data loader in cui passo il dataset che a cui verrà modificata la transform

train_loader_b, test_loader_b = utils.mnist_sum_2_loader(
    batch_size_train, batch_size_test, "basic",
    train_dataset=train_dataset_nosym, test_dataset=test_dataset_nosym
)
train_loader_b0, test_loader_b0 = utils.mnist_sum_2_loader(
    batch_size_train, batch_size_test, "mnistb0",
    train_dataset=train_dataset_nosym, test_dataset=test_dataset_nosym
)
train_loader_b3, test_loader_b3 = utils.mnist_sum_2_loader(
    batch_size_train, batch_size_test, "mnistb3",
    train_dataset=train_dataset_nosym, test_dataset=test_dataset_nosym
)

#classificatori
basic_sym = utils.MNISTNet_basic()
basic_nsym = utils.MNISTNet_basic()
b0_sym = utils.MNISTNet_b0()
b0_nsym = utils.MNISTNet_b0()
b3_sym = utils.MNISTNet_b3()
b3_nsym = utils.MNISTNet_b3()

#mnist sum net without sym
b_ns = utils.MNISTSum2Net(basic_nsym)
b0_ns = utils.MNISTSum2Net(b0_nsym)
b3_ns = utils.MNISTSum2Net(b3_nsym)

#mnist sum net with sym

b_s = utils.DeepProblogModel(basic_sym,  model_dir, data_dir, device,dataset_sym["basic"]["trainset"], dataset_sym["basic"]["testset"], batch_size_train, learning_rate, seed, method,cache=True)
b0_s = utils.DeepProblogModel(b0_sym, model_dir, data_dir, device, dataset_sym["b0"]["trainset"], dataset_sym["b0"]["testset"], batch_size_train, learning_rate, seed, method,cache=True)
b3_s = utils.DeepProblogModel(b3_sym,  model_dir, data_dir, device,dataset_sym["b3"]["trainset"], dataset_sym["b3"]["testset"], batch_size_train, learning_rate, seed, method,cache=True)

models = {"basic_neural":b_ns, "basic_nesy":b_s, 
        "b0_neural":b0_ns, "b0_nesy":b0_s, 
        "b3_neural":b3_ns, "b3_nesy":b3_s}

#trainer no sym
trainer_NoSym_basic = utils.Trainer_NoSym(train_loader_b, test_loader_b, model_dir,learning_rate, b_ns, device)
trainer_NoSym_b0 = utils.Trainer_NoSym(train_loader_b0, test_loader_b0, model_dir,learning_rate, b0_ns , device)
trainer_NoSym_b3 = utils.Trainer_NoSym(train_loader_b3, test_loader_b3, model_dir,learning_rate, b3_ns , device)

#training 
if no_train is True:
    load_all_models(model_dir, models, device)
    with open(data_dir+"/training_results.json", "r", encoding="utf-8") as f:
        training_results = json.load(f)
else:
    print("Inizio training dei modelli")
    
    t0_b_s = time.perf_counter() 
    rb_train_s = b_s.train(n_epochs,test_loader_singledigit_basic)
    print(f"Training modello basic nesy terminato in: {utils.time_delta_now(t0_b_s)}")

    t0_b_ns = time.perf_counter() 
    rb_train_ns = trainer_NoSym_basic.train(n_epochs)
    print(f"Training modello basic neural terminato in: {utils.time_delta_now(t0_b_ns)}")

    t0_b0_ns = time.perf_counter() 
    rb0_train_ns = trainer_NoSym_b0.train(n_epochs)
    print(f"Training modello b0 neural terminato in: {utils.time_delta_now(t0_b0_ns)}")

    t0_b0_s = time.perf_counter() 
    rb0_train_s = b0_s.train(n_epochs,test_loader_singledigit_b0)
    print(f"Training modello b0 nesy terminato in: {utils.time_delta_now(t0_b0_s)}")

    t0_b3_ns = time.perf_counter() 
    rb3_train_ns = trainer_NoSym_b3.train(n_epochs)
    print(f"Training modello b3 neural terminato in: {utils.time_delta_now(t0_b3_ns)}")

    t0_b3_s = time.perf_counter() 
    rb3_train_s = b3_s.train(n_epochs,test_loader_singledigit_b3)
    print(f"Training modello b3 nesy terminato in: {utils.time_delta_now(t0_b3_s)}")
    
    
    
    training_results = {
        "MNISTNet_basic": {
            "train_nesy": rb_train_s,
            "train_neural": rb_train_ns,
        },
        "EfficientNet-B0": {
            "train_nesy": rb0_train_s,
            "train_neural": rb0_train_ns,
        },
        "EfficientNet-B3": {
            "train_nesy": rb3_train_s,
            "train_neural": rb3_train_ns,
        },
    }   
    saveJsonResults("training_results", training_results)
    



#testing
print("Inizio testing dei modelli")
rb_test_s = b_s.test(test_loader_b)
rb0_test_s = b0_s.test(test_loader_b0)
rb3_test_s = b3_s.test(test_loader_b3)
rb_test_ns = trainer_NoSym_basic.test()
rb0_test_ns = trainer_NoSym_b0.test()
rb3_test_ns = trainer_NoSym_b3.test()

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
    if loss is None:
        return f"L:N.D.\nA:{acc*100:.1f}%\nD:{d1*100:.1f}/{d2*100:.1f}%"
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
    
    compGraph({"nesy": r["train_nesy"]["loss"], "neural": r["train_neural"]["loss"]},
            modelname,
            data_dir
            )
    
    df.loc[model, "Train (NeSy)"] = fmt_cell(min(r["train_nesy"]["loss"]), r["train_nesy"]["accuracy"], r["train_nesy"]["single-digit accuracy"])
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