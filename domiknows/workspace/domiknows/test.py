import sys
sys.path.append('../../../')
import json
import logging
logging.basicConfig(level=logging.INFO)

from data import get_readers
import torch
import numpy as np 
import random
from tqdm import tqdm
from sklearn.metrics import classification_report
from domiknows.program import IMLProgram, SolverPOIProgram
from domiknows.program.callbackprogram import hook
from domiknows import setProductionLogMode
import os
import argparse
import time
from model import build_program, NBSoftCrossEntropyLoss
import config

# build configs from command line args
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, choices=['Sampling', 'PrimalDual'], default='PrimalDual', help='Method of integrating constraints')
parser.add_argument('--model', type=str, choices=['basic', 'b0', 'b3'])
parser.add_argument("--modeldir", type=str, default="./model/mnist_sum_2")
parser.add_argument("--datadir", type=str, default="./data")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument('--log', type=str, default='None', choices=['None', 'TimeOnly', 'All'], help='None: no logs, TimeOnly: only output timing logs, All: output all logs. Logs will be found in the logs directory.')
parser.add_argument('--ILP', default=False, action='store_true', help='Use ILP during inference')
parser.add_argument('--no_fixedL', default=False, action='store_true', help='Don\'t fix summation labels in model. Use this flag when testing ILP.')

args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(args)

random.seed(args.seed); np.random.seed(args.seed)
torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
method = args.method
modeldir = args.modeldir
datadir = args.datadir
modelname = args.model


if args.log == 'None':
    setProductionLogMode(no_UseTimeLog=True)
elif args.log == 'TimeOnly':
    setProductionLogMode(no_UseTimeLog=False)

# import data
trainloader, trainloader_mini, validloader, testloader = get_readers(0, modelname=modelname)


def get_pred_from_node(node, suffix):
    pair_node = node.findDatanodes(select='pair')[0]
    digit0_node = node.findDatanodes(select='image')[0]
    digit1_node = node.findDatanodes(select='image')[1]

    #print(digit0_node.getAttributes())

    if torch.cuda.is_available():
        digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}')).cpu()
        digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}')).cpu()
        #summation_pred = torch.argmax(pair_node.getAttribute(f'<summations>{suffix}')).cpu()
    else:
        digit0_pred = torch.argmax(digit0_node.getAttribute(f'<digits>{suffix}'))
        digit1_pred = torch.argmax(digit1_node.getAttribute(f'<digits>{suffix}'))
        #summation_pred = torch.argmax(pair_node.getAttribute(f'<summations>{suffix}'))

    summation_pred = digit0_pred + digit1_pred

    return digit0_pred, digit1_pred, summation_pred


#program.populate(reader, device='auto')

def get_classification_report(program, reader, total=None, verbose=False, infer_suffixes=['/local/argmax'], print_incorrect=False):
    digits_results = {
        'label': []
    }

    summation_results = {
        'label': []
    }

    satisfied = {}

    satisfied_overall = {}

    for suffix in infer_suffixes:
        digits_results[suffix] = []
        summation_results[suffix] = []

    # iter through test data
    for i, node in enumerate(program.populate(reader, device=device)):

        # e.g. /local/argmax or /ILP
        for suffix in infer_suffixes:
            # get predictions and add to list
            digit0_pred, digit1_pred, summation_pred = get_pred_from_node(node, suffix)

            digits_results[suffix].append(digit0_pred)
            digits_results[suffix].append(digit1_pred)

            summation_results[suffix].append(summation_pred)

        # get labels and add to list
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

        if print_incorrect and (digits_results['/local/argmax'][-1] != digits_results['label'][-1] or digits_results['/local/argmax'][-2] != digits_results['label'][-2]):
            for suffix in infer_suffixes:
                print("%s: %d + %d = %d" % (suffix,
                                        digits_results[suffix][-1],
                                        digits_results[suffix][-2],
                                        summation_results[suffix][-1]))
                print("real: %d + %d = %d" % (digits_results['label'][-1],
                                        digits_results['label'][-2],
                                        summation_results['label'][-1]))
            print()

        # get constraint verification stats
        for suffix in infer_suffixes:
            verifyResult = node.verifyResultsLC(key=suffix)
            if verifyResult:
                satisfied_constraints = []
                ifSatisfied_avg = 0.0
                ifSatisfied_total = 0
                for lc_idx, lc in enumerate(verifyResult):
                    # add constraint satisfaction to total list (across all samples)
                    if lc not in satisfied:
                        satisfied[lc] = []
                    satisfied[lc].append(verifyResult[lc]['satisfied'])
                    satisfied_constraints.append(verifyResult[lc]['satisfied'])

                    # build average ifSatisfied value for this single sample
                    if 'ifSatisfied' in verifyResult[lc]:
                        if verifyResult[lc]['ifSatisfied'] == verifyResult[lc]['ifSatisfied']:
                            ifSatisfied_avg += verifyResult[lc]['ifSatisfied']
                            ifSatisfied_total += 1

                # add average ifSatisifed value to overall stats
                if suffix not in satisfied_overall:
                    satisfied_overall[suffix] = []

                satisfied_overall[suffix].append(ifSatisfied_avg / ifSatisfied_total)

                #satisfied_overall[suffix].append(1 if num_constraints * 100 == sum(satisfied_constraints) else 0)
                #pred_digit_sum = digits_results[suffix][-1] + digits_results[suffix][-2]
                #label_sum = summation_results['label'][-1]
                #satisfied_overall[suffix].append(1 if pred_digit_sum == label_sum else 0)
    results = {suf:{'accuracy':0.0, 'single-digit accuracy':0.0} for suf in infer_suffixes}
    for suffix in infer_suffixes:
        print('============== RESULTS FOR:', suffix, '==============')

        if verbose:
            for j, (digit_pred, digit_gt) in enumerate(zip(digits_results[suffix], digits_results['label'])):
                print(f'digit {j % 2}: pred {digit_pred}, gt {digit_gt}')

                if j % 2 == 1:
                    print(f'summation: pred {summation_results[suffix][j // 2]},'
                          f'gt {summation_results["label"][j // 2]}\n')


        sdacc= classification_report(digits_results['label'], digits_results[suffix], output_dict=True)['accuracy']
        results[suffix]['accuracy'] = classification_report(summation_results['label'], summation_results[suffix], output_dict=True)['accuracy']
        results[suffix]['single-digit accuracy'] = (sdacc,sdacc)
        print(f"sd-acc: {sdacc*100:.2f}%, sum_acc: {results[suffix]['accuracy']*100:.2f}% in {suffix}")
        #print(classification_report(digits_results['label'], digits_results[suffix], digits=5))
        #print(classification_report(summation_results['label'], summation_results[suffix], digits=5))

        print('==========================================')

    #sat_values = list(chain(*satisfied.values()))
    #print('Average constraint satisfactions: %f' % (sum(sat_values)/len(sat_values)))]
    for suffix in infer_suffixes:
        suffixSatisfiedNumpy = np.array(satisfied_overall[suffix])
        print('Average constraint satisfactions: %s - %f' % (suffix, np.nanmean(suffixSatisfiedNumpy)))
    
    return results


use_digit_labels = False

sum_setting = None

graph, image, image_pair, image_batch = build_program(modeltype=modelname,device=device, sum_setting=sum_setting, digit_labels=use_digit_labels, use_fixedL= not args.no_fixedL, test=True)


inferTypes = ['local/argmax']
if args.ILP:
    inferTypes.append('ILP')

program = SolverPOIProgram(graph,
                            poi=(image_batch, image, image_pair),
                            inferTypes=inferTypes,
                            metric={})


pt_files = [f for f in os.listdir(args.modeldir) if f.endswith('.pt')]
    #device="cpu"
find = False
for f in pt_files:
    filename_pt = f.lower()
    filename = (f"{modelname}_{method}").lower()
    if filename in filename_pt:
        path = args.modeldir + "/" + f
        state = torch.load(path, map_location=device)
        program.model.load_state_dict(state)
        print(f"modello caricato per {modelname}")
        find = True
        break
if not find:
    print(f"Attenzione, modello {modelname} pre-saved non trovato")

classification_suffixes = ['/local/argmax']
if args.ILP:
    classification_suffixes.append('/ILP')

#print("test constraint satisfaction")
#program.verifyResultsLC(testloader, device=device)

# get test accuracy
print("test evaluation")
results = get_classification_report(program, testloader, total=config.num_test, verbose=False, infer_suffixes=classification_suffixes, print_incorrect=False)
save_path = os.path.join(args.datadir, f"{modelname}/{method}/test_metrics.json")
with open(save_path, 'w') as f:
    json.dump(results, f, indent=2)
