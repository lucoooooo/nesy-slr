
import matplotlib.pyplot as plt #
from matplotlib.table import Table
import pandas as pd
import argparse
import os
import json
import pandas as pd

def compGraph(accuracy, modelname, data_dir):
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

def get_result_nesy(datadir, filename, method = None):
    suffixes = ['basic','b0','b3']
    results = {suf: {} for suf in suffixes}
    for name in suffixes:
        if method is None: path = f"{name}/{filename}"
        else: path = f"{name}/{method}/{filename}"
        try:
            with open(os.path.join(datadir,path),"r") as f:
                results[name] = json.load(f)
        except Exception:
            print(f"Nessun file per {path}")
    return results
parser = argparse.ArgumentParser()

# Parameters

parser.add_argument("--resultdir", type=str, default="./results")
parser.add_argument("--nesy_datadir", type=str, default="./domiknows/data")
parser.add_argument("--neural_datadir", type=str, default="./data")
parser.add_argument('--method', type=str, choices=['Sampling', 'Semantic', 'PrimalDual'], default='PrimalDual', help='Method used')

args = parser.parse_args()
resdir = args.resultdir

os.makedirs(resdir, exist_ok=True)
os.makedirs(os.path.join(resdir,"basic"),exist_ok=True)
os.makedirs(os.path.join(resdir,"b0"), exist_ok=True)
os.makedirs(os.path.join(resdir,"b3"), exist_ok=True)

training_nesy = get_result_nesy(args.nesy_datadir,"training_metrics.json",args.method)
with open(args.neural_datadir+"/training_results.json", "r", encoding="utf-8") as f:
        training_neural = json.load(f)

#finire di aggiustare questo
testing_nesy = get_result_nesy(args.nesy_datadir,"test_metrics.json",args.method)
with open(args.neural_datadir+"/testing_results.json", "r", encoding="utf-8") as f: 
        testing_neural = json.load(f)

results = {
    "MNISTNet_basic": {
        "train_nesy": training_nesy["basic"],
        "test_nesy":  testing_nesy["basic"]["/local/argmax"],
        "train_neural": training_neural["MNISTNet_basic"]["train_neural"],
        "test_neural": testing_neural["MNISTNet_basic"]["test_neural"],
    },
    "EfficientNet-B0": {
        "train_nesy":  training_nesy["b0"],
        "test_nesy":   testing_nesy["b0"]["/local/argmax"],
        "train_neural": training_neural["EfficientNet-B0"]["train_neural"],
        "test_neural": testing_neural["EfficientNet-B0"]["test_neural"],
    },
    "EfficientNet-B3": {
        "train_nesy":  training_nesy["b3"],
        "test_nesy":   testing_nesy["b3"]["/local/argmax"],
        "train_neural": training_neural["EfficientNet-B3"]["train_neural"],
        "test_neural": testing_neural["EfficientNet-B3"]["test_neural"],
    },
}
"""
prova = {"MNISTNet_basic": {
        "train_nesy": training_nesy["basic"],
        "test_nesy":  testing_nesy["basic"]["/local/argmax"],
        "train_neural": training_neural["MNISTNet_basic"]["train_neural"],
        "test_neural": testing_neural["MNISTNet_basic"]["test_neural"],
    }}
"""
# salvataggio risultati

rows = list(results.keys())
#[debug]rows = list(prova.keys())
cols = ["Train (NeSy)", "Test (NeSy)","Train (Neural)", "Test (Neural)"]

def fmt_cell(loss, acc, single_acc):
    d1, d2 = single_acc
    """DEBUG
    if isinstance(single_acc,tuple):
        d1, d2 = single_acc
    elif isinstance(single_acc,list):
        d1 = max(single_acc)
        d2 = d1
    else:
        d1=single_acc
        d2=d1
    """
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
    compGraph(
            {"nesy": r["train_nesy"]["accuracy"], "neural": r["train_neural"]["accuracy"]},
            modelname,
            resdir
            )
    
    df.loc[model, "Train (NeSy)"] = fmt_cell(None, max(r["train_nesy"]["accuracy"]), r["train_nesy"]["single-digit accuracy"])
    df.loc[model, "Test (NeSy)"] = fmt_cell(None, r["test_nesy"]["accuracy"], r["test_nesy"]["single-digit accuracy"])
    df.loc[model, "Train (Neural)"] = fmt_cell(min(r["train_neural"]["loss"]), max(r["train_neural"]["accuracy"]), r["train_neural"]["single-digit accuracy"])
    df.loc[model, "Test (Neural)"] = fmt_cell(r["test_neural"]["loss"], r["test_neural"]["accuracy"], r["test_neural"]["single-digit accuracy"])

def save_table_image(df : pd.DataFrame, out_png=f"{resdir}/results.png", figsize=(12, 4.5), dpi=200, header_color="#f0f0f0"):
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