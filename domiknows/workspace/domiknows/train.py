import sys
sys.path.append('../../../')

import logging
logging.basicConfig(level=logging.INFO)

from data import get_readers
import torch
import random
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from domiknows.program.lossprogram import PrimalDualProgram, SampleLossProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows import setProductionLogMode
import json
import os
import argparse

from model import build_program
import config



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


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, choices=['Sampling', 'PrimalDual'], help='Method of integrating constraints')
parser.add_argument('--num_train', type=int, default=500, help='Number of training iterations per epoch')
parser.add_argument('--log', type=str, default='None', choices=['None', 'TimeOnly', 'All'], help='None: no logs, TimeOnly: only output timing logs, All: output all logs. Logs will be found in the logs directory.')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
parser.add_argument('--model', type=str, choices=['basic', 'b0', 'b3'])
parser.add_argument("--modeldir", type=str, default="./model/mnist_sum_2")
parser.add_argument("--datadir", type=str, default="./data")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument('--warmup', type=int, default=0)


args = parser.parse_args()
modelname = args.model
lr = args.lr
batch_size = args.batch_size
print(args)

warmup = args.warmup
method = args.method
num_train = args.num_train
os.makedirs(args.modeldir, exist_ok=True)
os.makedirs(args.datadir, exist_ok=True)
os.makedirs(os.path.join(args.datadir,f"{modelname}/{method}"),exist_ok=True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.log == 'None':
    setProductionLogMode(no_UseTimeLog=True)
elif args.log == 'TimeOnly':
    setProductionLogMode(no_UseTimeLog=False)

trainloader, trainloader_mini, validloader, testloader = get_readers(num_train, modelname)


def get_pred_from_node(node, suffix):
    digit0_node = node.findDatanodes(select='image')[0]
    digit1_node = node.findDatanodes(select='image')[1]

    if torch.cuda.is_available():
        digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}')).cpu()
        digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}')).cpu()
    else:
        digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}'))
        digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}'))

    #summation_pred = torch.argmax(node.getAttribute(f'<summations>{suffix}'))
    summation_pred = digit0_pred + digit1_pred
    return digit0_pred, digit1_pred, summation_pred



def get_classification_report(program, reader, total=None, verbose=False, infer_suffixes=['/local/argmax']):
    digits_results = {
        'label': []
    }

    summation_results = {
        'label': []
    }

    for suffix in infer_suffixes:
        digits_results[suffix] = []
        summation_results[suffix] = []

    for i, node in tqdm(enumerate(program.populate(reader, device=device)), total=total, position=0, leave=True):

        for suffix in infer_suffixes:
            digit0_pred, digit1_pred, summation_pred = get_pred_from_node(node, suffix)

            digits_results[suffix].append(digit0_pred.cpu().item())
            digits_results[suffix].append(digit1_pred.cpu().item())

            summation_results[suffix].append(summation_pred)

        pair_node = node.findDatanodes(select='pair')[0]
        digit0_node = node.findDatanodes(select='image')[0]
        digit1_node = node.findDatanodes(select='image')[1]

        if torch.cuda.is_available():
            digits_results['label'].append(digit0_node.getAttribute('digit_label').cpu().item())
            digits_results['label'].append(digit1_node.getAttribute('digit_label').cpu().item())
            summation_results['label'].append(pair_node.getAttribute('summation_label').cpu().item())
        else:
            digits_results['label'].append(digit0_node.getAttribute('digit_label').item())
            digits_results['label'].append(digit1_node.getAttribute('digit_label').item())
            summation_results['label'].append(pair_node.getAttribute('summation_label'))

    sum_acc = 0.0
    digit_acc = 0.0
    for suffix in infer_suffixes:
        print('============== RESULTS FOR:', suffix, '==============')

        if verbose:
            for j, (digit_pred, digit_gt) in enumerate(zip(digits_results[suffix], digits_results['label'])):
                print(f'digit {j % 2}: pred {digit_pred}, gt {digit_gt}')

                if j % 2 == 1:
                    print(f'summation: pred {summation_results[suffix][j // 2]},'
                          f'gt {summation_results["label"][j // 2]}\n')

        #print(classification_report(digits_results['label'], digits_results[suffix], digits=5))
        #print(classification_report(summation_results['label'], summation_results[suffix], digits=5))
        digit_acc = classification_report(digits_results['label'], digits_results[suffix], output_dict=True)['accuracy']
        sum_acc = classification_report(summation_results['label'], summation_results[suffix], output_dict=True)['accuracy']
        print("DIGIT ACC: \t",digit_acc)
        print("SOMMA: \t",sum_acc)
        #print(classification_report(digits_results['label'], digits_results[suffix]))
        #print(classification_report(summation_results['label'], summation_results[suffix]))

    return sum_acc,digit_acc
        


use_digit_labels = (method == 'DigitLabel')

sum_setting = None



graph, image, image_pair, image_batch = build_program(modeltype=modelname,device=device, sum_setting=sum_setting, digit_labels=use_digit_labels)

if method == 'PrimalDual':
    class PrimalDualCallbackProgram(PrimalDualProgram):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.after_train_epoch = []

        def call_epoch(self, name, dataset, epoch_fn, **kwargs):
            if name == 'Testing':
                for fn in self.after_train_epoch:
                    fn(kwargs)
            else:
                super().call_epoch(name, dataset, epoch_fn, **kwargs)


    program = PrimalDualCallbackProgram(graph, SolverModel,
                        poi=(image_batch, image, image_pair),
                        inferTypes=['local/argmax'],
                        metric={})
                        #batch_size=batch_size)

elif method == 'Sampling':
    class CallbackProgram(SampleLossProgram):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.after_train_epoch = []

        def call_epoch(self, name, dataset, epoch_fn, **kwargs):
            if name == 'Testing':
                for fn in self.after_train_epoch:
                    fn(kwargs)
            else:
                super().call_epoch(name, dataset, epoch_fn, **kwargs)


    program = CallbackProgram(graph, SolverModel,
                            poi=(image_batch, image, image_pair),
                            inferTypes=['local/argmax'],
                            metric={},
                            sample=True,
                            sampleSize=100,
                            sampleGlobalLoss=True,
                            beta=1)
                            #batch_size=batch_size)



epoch_num = 1
sum_accs = []
digit_accs= []


def post_epoch_metrics(kwargs, interval=1, train=True):
    if epoch_num % interval == 0:
        if train:
            sum_acc,digit_acc = get_classification_report(
                program, trainloader_mini, total=config.num_valid, verbose=False
            )
            sum_accs.append(sum_acc)
            digit_accs.append(digit_acc)
            sdacc = max(digit_accs)
            epoch_results = {
                "loss": None,
                "accuracy": sum_accs,
                "single-digit accuracy": (sdacc,sdacc)
            }
            
            save_path = os.path.join(args.datadir, f"{modelname}/{method}/training_metrics.json") #farò solo PrimalDual
            with open(save_path, 'w') as f:
                json.dump(epoch_results, f, indent=2)


program.after_train_epoch = [post_epoch_metrics]

t0 = time.perf_counter()

if method == 'Sampling':
    def test_adam(params):
        print('initializing optimizer')
        return torch.optim.Adam(params, lr=5e-4)


    program.train(trainloader,
                  train_epoch_num=args.epochs,
                  Optim=test_adam,
                  device=device,
                  test_every_epoch=True,
                  c_warmup_iters=warmup)
                  #batch_size=batch_size)

elif method == 'PrimalDual':
    def test_adam(params):
        print('initializing optimizer')
        return torch.optim.Adam(params, lr=lr)


    program.train(trainloader,
                  train_epoch_num=args.epochs,
                  Optim=test_adam,
                  device=device,
                  test_every_epoch=True,
                  c_warmup_iters=warmup)
                  #batch_size=batch_size

print(f"Training modello {modelname} nesy terminato in: {time_delta_now(t0)}")
torch.save(program.model.state_dict(), os.path.join(args.modeldir, f"{modelname}_{method}_nesy.pt"))

