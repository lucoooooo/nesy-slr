import json
import os
import random
from typing import *
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from sklearn.metrics import accuracy_score
from abc import ABC, abstractmethod
import time
from NSIL.src.NSILDataset import NSILDataset, NSILImageLoader
from NSIL.src.ASPUtils import find_all_stable_models, batch_run_evaluation
from NSIL.src.NSILTask.base import NSILTask
from NSIL.src.ILP.WCDPI import WCDPI
import re
from NSIL.src.NSILNetworkConfiguration import NSILNetworkConfiguration
from NSIL.src.NSILRun import NSILRun
from NSIL.global_config import CustomArgument, set_random_seeds
import numpy as np
from NSIL.src.ILP.WCDPI import DefaultWCDPIPair
from NSIL.src.NSIL import NSIL

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

    return f"{hours} hours, {minutes} minutes, {seconds} seconds, {millisecs} milliseconds"



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
    #torchvision.transforms.Normalize(mean=(0.1307,)*3,
    #                    std=(0.3081,)*3),
    torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],
                                    std=[0.229,0.224,0.225]),
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
        seed : int = None
    ):
        # Contains a MNIST dataset
        self.mnist_dataset = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
        )
        self.dataset_name = "train" if train else "test"
        self.seed=seed
        self.index_map = list(range(len(self.mnist_dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.index_map)
        else:
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


class ArithmeticDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        train = False,
        seed : int = None
    ):
        # Contains a MNIST dataset
        self.mnist_dataset = dataset
        self.dataset_name = "train" if train else "test"
        self.seed=seed
        self.index_map = list(range(len(self.mnist_dataset)))
        self.unique_labels = list(range(19))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(self.index_map)
        else:
            random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.index_map) / 2)

    def __getitem__(self, idx):
        idx1 = self.index_map[idx * 2]
        idx2 = self.index_map[idx * 2 + 1]
        (a_img, a_digit) = self.mnist_dataset[idx1]
        (b_img, b_digit) = self.mnist_dataset[idx2]
        summation = a_digit+b_digit

        return [idx1, idx2], summation, [a_img, b_img]
    @staticmethod 
    def arithmetic_collate_fn(batch):
        
        x_idxs_list, labels_list, images_list = zip(*batch)

        labels = torch.tensor(labels_list)
        
        x_idxs = torch.tensor(x_idxs_list)
        num_images_per_sample = len(images_list[0])
        stacked_images = []
        for i in range(num_images_per_sample):
            img_pos_i = torch.stack([sample[i] for sample in images_list])
            stacked_images.append(img_pos_i)

        return x_idxs, labels, stacked_images

#rimodellato per dataset dinamico e non csv
class MNISTSum2NSIL(NSILDataset):
    def __init__(self, runner,
                data_dir,
                transform, 
                seed,
                num_workers,
                batch_size):
        runner.logger.info('load_data')
        self.args = runner.args
        self.num_w = num_workers
        self.seed = seed
        self.batch_size = batch_size
        self.image_data = {
            'train': torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform),
            'test': torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        }
        
        self.neurasp = ArithmeticDataset(self.image_data['train'], self.seed)
        if hasattr(self.args, 'pct') and self.args.pct < 100:
            total_len_train = len(self.neurasp.index_map)
            keep_len_train = int(total_len_train * (self.args.pct / 100.0))
            self.neurasp.index_map = self.neurasp.index_map[:keep_len_train]
        super().__init__(runner, data_dir)

    def load_nesy_data(self):
        train_data = ArithmeticDataset(self.image_data['train'], True, self.seed)
        if hasattr(self.args, 'pct') and self.args.pct < 100:
            total_len_train = len(train_data.index_map)
            keep_len_train = int(total_len_train * (self.args.pct / 100.0))
            train_data.index_map = train_data.index_map[:keep_len_train]
        self.index_map_train = train_data.index_map
        val_data = ArithmeticDataset(self.image_data['train'], self.seed)
        val_data.index_map = train_data.index_map
        test_data = ArithmeticDataset(self.image_data['test'], self.seed)
        if hasattr(self.args, 'pct') and self.args.pct < 100:
            total_len_test = len(test_data.index_map)
            keep_len_test = int(total_len_test * (self.args.pct / 100.0))
            test_data.index_map = test_data.index_map[:keep_len_test]
        
        train_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=self.num_w, collate_fn=ArithmeticDataset.arithmetic_collate_fn)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=self.num_w, collate_fn=ArithmeticDataset.arithmetic_collate_fn)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, num_workers=self.num_w, collate_fn=ArithmeticDataset.arithmetic_collate_fn)
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def load_nn_data(self):
        used_indices = self.index_map_train

        train_images, train_labels, train_idx_map = [], [], []
        base_mnist = self.image_data['train']

        for idx in used_indices:
            img, label = base_mnist[idx]
            train_images.append(img)
            train_labels.append(label)
            train_idx_map.append(idx)

        new_train_ds = NSILImageLoader(train_images, train_labels, train_idx_map)
        
        return {
            'digit': {
                'train': DataLoader(new_train_ds, batch_size=self.batch_size, num_workers=self.args.num_workers),
                'test': DataLoader(self.image_data['test'], batch_size=self.batch_size, num_workers=self.args.num_workers)
            }
        }
        

    def convert_to_NeurASP(self):
        """
        Convert data into data_list and obs_list for NeurASP training
        """
        data = self.neurasp
        data_list = []
        obs_list = []
        for item in data:
            d_list_entry = {}
            indices = item[0]
            label = item[1]
            for idx, image_idx in enumerate(indices):
                i = f'i{idx + 1}'
                d_list_entry[i] = self.image_data['train'][image_idx][0]
            
            data_list.append(d_list_entry)
            obs_list.append(f':- not result({label}).')
            
        return data_list, obs_list

#fix per batch altrimenti mi sballa i risultati di sum_Acc e h_Acc
def batched_downstream_task_eval(self, nn_preds, ground_preds, data_type='test'):
    nn_preds_correct = 0
    ground_preds_correct = 0
    total = 0
    flat_pred_idx = 0
    
    # itero sui batch
    for batch in self.task.nesy_data[data_type]:
        batch_labels = batch[1] # batchsize esempi
        
        # itero per esempio
        for i in range(len(batch_labels)):
            label = int(batch_labels[i].item())
            
            # evaluation
            if flat_pred_idx < len(nn_preds):
                if label == nn_preds[flat_pred_idx]:
                    nn_preds_correct += 1
                if label == ground_preds[flat_pred_idx]:
                    ground_preds_correct += 1
            
            flat_pred_idx += 1
            total += 1
            
    if total == 0: return 0.0, 0.0
    
    nn_acc = nn_preds_correct / total
    ground_acc = ground_preds_correct / total
    
    return nn_acc, ground_acc

#loaded from examples
class ArithmeticTask(NSILTask):
    def __init__(self, data, runner, net_confs, image_to_net_map, digit_vals, ilp_config=None, n_digits=2):
        """
        Arithmetic Tasks
        @param data: NSILDataset
        @param runner: NSILRun instance
        @param net_confs: dict of NSIL neural network configurations
        @param image_to_net_map: image to network map
        @param digit_vals: possible digit values (e.g. [0,1,...,9])
        @param ilp_config: custom bk and md
        @param n_digits: number of digits in each example
        """
        self.digit_vals = digit_vals
        self.n_digits = n_digits
        if ilp_config:
            bk = ilp_config.bk
            md = ilp_config.md
        else:
            bk = '''
            :- digit(1,X0), digit(2,X1), result(Y1), result(Y2), Y1 != Y2.
            result(Y) :- digit(1,X0), digit(2,X1), solution(X0,X1,Y).
            num(0..18).
            digit_type(0..9).
            '''

            md = f'''
            #modeh(solution(var(digit_type),var(digit_type),var(num))).
            #modeb(var(num) = var(digit_type)).
            #modeb(var(num) = var(digit_type) + var(digit_type)).
            #maxv(3).
            
            #bias("penalty(1, head(X)) :- in_head(X).").
            #bias("penalty(1, body(X)) :- in_body(X).").
            '''

        # Remove spaces introduced by multi-line string
        def clean(x): return '\n'.join([l.strip() for l in x.split('\n')])

        md = clean(md)
        bk = clean(bk)
        super().__init__(data, runner, net_confs, image_to_net_map, md, bk)

    def _create_n_choice_rules(self, n):
        """
        Create choice rules for each digit fact
        @param n: number of digits
        @return: choice rules
        """
        cr = []
        for d in range(n):
            _d_rule = '1 {'
            _d_rule += '; '.join([f'digit({d + 1},{v})' for v in self.digit_vals])
            _d_rule = f'{_d_rule} }} 1.'
            cr.append(_d_rule)
        return cr

    def create_bootstrap_task(self):
        # For each unique_label in the dataset, create LAS example with choice rules for possible digit values
        ex_id = 0
        poss_labels = self.nesy_data['train'].dataset.unique_labels
        for pl in poss_labels:
            self.bootstrap_examples[f'bootstrap_{ex_id}'] = WCDPI(
                ex_id=f'bootstrap_{ex_id}',
                positive=True,
                weight='inf',
                inclusion=[f'result({pl})'],
                exclusion=[],
                context=self._create_n_choice_rules(self.n_digits)
            )
            ex_id += 1

        # Create final task
        ex_str = "\n".join([str(self.bootstrap_examples[e]) for e in self.bootstrap_examples])
        return f'{ex_str}\n{self.bk}\n{self.md}'

    def convert_las_hyp_to_nesy_r(self, h):
        """
        Convert a LAS hypothesis into a NeurASP representation
        @param h: the hypothesis
        """
        converted_h = h
        self.logger.info('converted_h', converted_h)
        return converted_h

    def model_to_network_preds(self, m):
        m = sorted(m.split(' '))
        # Extract network predictions from stable model
        net_preds = [int(f.split(f'digit({idx + 1},')[1].split(')')[0]) for idx, f in enumerate(m)]
        return net_preds

    def get_combo_id(self, ex, for_ilp=True):
        label = ex[1][0].item()
        return f'label_{label}'

    def get_context_facts(self, poss):
        facts = ''
        for idx, n in enumerate(poss):
            if type(n) == torch.Tensor:
                n = int(n.item())
            facts += f'digit({idx+1},{int(n)}). '
        facts = facts[:-1]
        return facts

    def compute_stable_models(self, asp_prog, obs, num_images, for_ilp=True):
        # Create choice rules
        cr = self._create_n_choice_rules(self.n_digits)
        cr = '\n'.join(cr)
        program = f'{cr}\n{asp_prog}\n{obs}\n#show digit/2.'

        # Build ID and load from cache if saved
        label = obs.split('result(')[1].split(')')[0]
        combo_id = f'label_{label}'
        prog_id = re.sub(r"[\n\t\s]*", "", asp_prog)
        if prog_id in self.stable_model_cache and combo_id in self.stable_model_cache[prog_id]:
            return self.stable_model_cache[prog_id][combo_id]

        # Otherwise, call clingo
        models = find_all_stable_models(program, self.model_to_network_preds)
        if len(models) == 0:
            print('ERROR: 0 stable models for program:')
            print(program)
            print('--------')

        # Save to cache
        if prog_id not in self.stable_model_cache:
            self.stable_model_cache[prog_id] = {combo_id: models}
        else:
            self.stable_model_cache[prog_id][combo_id] = models
        return models

    def symbolic_evaluation(self, i, latent_concepts, h, data_type='test', preds_type='nn'):
        self.logger.info('start_symbolic_eval', data_type, preds_type)
        start_time = time.time()
        lc = latent_concepts['digit']['predictions']
        idx_dict = {}
        if 'idx_map' in latent_concepts['digit'] and latent_concepts['digit']['idx_map'] is not None:
            idx_map = latent_concepts['digit']['idx_map']
            idx_dict = {global_id: pos_idx for pos_idx, global_id in enumerate(idx_map)}

        header = f'{self.bk}\n{h}'
        footer = '#show result/1.'
        examples = []

        for ex in self.nesy_data[data_type]:
            batch_indices = ex[0]
            for j in range(len(batch_indices)):
                sample_idxs = batch_indices[j] 
                current_preds = []
                for idx_tensor in sample_idxs:
                    raw_idx = idx_tensor.item()
                    if data_type == 'train' and idx_dict and raw_idx in idx_dict:
                        final_idx = idx_dict[raw_idx]
                        current_preds.append(lc[final_idx])
                    elif data_type == 'test' and raw_idx < len(lc):
                        current_preds.append(lc[raw_idx])
                    else:
                        current_preds.append(0)
                facts = self.get_context_facts(current_preds)
                examples.append(facts)

        predictions = batch_run_evaluation(header, examples, footer)
        predictions = [int(predictions[key][0].split('result(')[1].split(')')[0]) for key in sorted(predictions.keys())]
        
        end_time = time.time() - start_time
        if data_type == 'train':
            self.logger.add_component_time(i, f'symbolic_{data_type}_{preds_type}_preds_eval', end_time)
        return predictions

    #generazione esempi correttivi -> calcola l'errore del modello logico e crea esempi che permettono di sbloccare una situazione idi stallo della rete (guida la rete)
    def calculate_train_FNR(self, downstream_train_preds):
        combo_count = {}
        combo_correct = {}
        
        flat_idx = 0 
        
        # itero sui batch
        for batch in self.nesy_data['train']:
            #batch di <indici,label,img>
            batch_labels = batch[1]
            # itero sull'esempio
            for i in range(len(batch_labels)):
                label = int(batch_labels[i].item())
                combo_id = f'label_{label}'
                if combo_id:
                    if combo_id not in combo_count:
                        combo_count[combo_id] = 0
                        combo_correct[combo_id] = 0
                    combo_count[combo_id] += 1
                    if flat_idx < len(downstream_train_preds):
                        pred = downstream_train_preds[flat_idx]
                        if label == pred:
                            combo_correct[combo_id] += 1
                flat_idx += 1
        fnr = {}
        for c in combo_count:
            total = combo_count[c]
            correct = combo_correct[c]
            if total > 0:
                fnr[c] = 100.0 * (1.0 - (correct / total))
            else:
                fnr[c] = 0.0
        return fnr

    def exploration(self, i, downstream_train_preds, h):
        start_time = time.time()
        self.logger.info('exploration_start')
        fnr = self.calculate_train_FNR(downstream_train_preds)
        self.logger.info('fnr', fnr)
        
        asp_prog = f'{self.bk}\n{h}'
        combos_done = []

        # explore aggiustate in modo da avere solo due pred digit
        for batch in self.nesy_data['train']:
            batch_indices = batch[0] #coppie di indici
            batch_labels = batch[1] #somme
            #print(f"batch IN EXPLORE: {batch}")
            
            # itero sui singoli esempi
            batch_size = len(batch_labels)
            for j in range(batch_size):
                
                label = batch_labels[j].item() #somma
                single_indices = batch_indices[j] #<img1,img2>
                #print(f" IN EXPLORE: <{single_indices},{label}>")
                combo_id = f'label_{label}'  
                if combo_id and combo_id not in combos_done:
                    # calcolo dei modelli stabili che diano come risultato solo label
                    possible_ctxs = self.compute_stable_models(
                        asp_prog, 
                        f':- not result({label}).', 
                        len(single_indices) 
                    )
                    #print(f"modelli stabili IN EXPLORE: {possible_ctxs}")
                    for poss in possible_ctxs:
                        # regola explore
                        facts = self.get_context_facts(poss)
                        #print(f"fatti in explore: {facts}")
                        ex_id = f'{combo_id}_' + '_'.join([str(p) for p in poss])
                        #print(f"ex_id in EXPLORE: {ex_id}")
                        if ex_id not in self.symbolic_examples:
                            self.symbolic_examples[ex_id] = DefaultWCDPIPair(
                                combo_id=ex_id,
                                label=label,
                                ctx_facts=[facts]
                            )
                        if len(possible_ctxs) > 0:
                            weight = fnr[combo_id] / len(possible_ctxs)
                            self.update_WCDPI_weights(ex_id, weight, fnr=True)
                    combos_done.append(combo_id)

        total_time = time.time() - start_time
        self.logger.add_component_time(i, 'exploration', total_time)


    #se explore si basa sul modello logico, questa si basa sul modello neurale (per correggere la logica)
    def exploitation(self, i, nn_output):
        start_time = time.time()
        self.logger.info('exploitation_start')
        
        predictions = nn_output['digit']['predictions']
        #print(f"PREDIZIONI IN EXPLOIT: {predictions} e shape: {predictions.shape}")
        conf_scores = nn_output['digit']['confidence']
        idx_map = nn_output['digit']['idx_map']
        idx_dict = {global_id: pos_idx for pos_idx, global_id in enumerate(idx_map)}
        #print(f"idx_map IN EXPLOIT: {idx_map}")
        combos = {}

        # altrimenti crea regole sbagliate
        for batch in self.nesy_data['train']:
            batch_indices = batch[0]
            batch_labels = batch[1]
            #print(f"batch IN EXPLOIT: {batch}")
            # ogni exploit ha due digit e non 32
            batch_size = len(batch_labels)
            for j in range(batch_size):
                label = batch_labels[j].item() #somma
                single_indices = batch_indices[j] #<img1,img2>
                #print(f"output reti IN EXPLOIT: <{single_indices},{label}>")
                #si controllano gli output neurali
                x = []
                confs = []
                for img_idx_tensor in single_indices:
                    global_idx = img_idx_tensor.item()
                    mapped_idx = idx_dict[global_idx]
                    val = int(predictions[mapped_idx].item())
                    conf = conf_scores[mapped_idx].item()
                    x.append(val)
                    confs.append(conf)

                #confidenza aggregata
                agg_conf = np.prod(np.array(confs))
                combo_id = f'label_{label}'
                
                if combo_id:
                    #id per i numeri predetti
                    full_combo_id = f'{combo_id}_' + '_'.join([str(c) for c in x])
                    #print(f"combo_ID in EXPLOIT:{full_combo_id}")
                    if full_combo_id in combos:
                        combos[full_combo_id]['weights'].append(agg_conf)
                    else:
                        start_list = self.get_context_facts(x)
                        combos[full_combo_id] = {
                            'weights': [agg_conf],
                            'context': [start_list],
                            'label': label
                        }
        for combo_id in combos:
            weight = 100 * np.mean(np.array(combos[combo_id]['weights']))
            if combo_id not in self.symbolic_examples:
                label = combos[combo_id]['label']
                ctx = combos[combo_id]['context']
                self.symbolic_examples[combo_id] = DefaultWCDPIPair(
                    combo_id=combo_id, 
                    label=label, 
                    ctx_facts=ctx
                )
            #la confidenza viene usata come peso per aggiornare le regole logiche e guidare il solver las
            self.update_WCDPI_weights(combo_id, weight, fnr=False)
        total_time = time.time() - start_time
        self.logger.add_component_time(i, 'exploitation', total_time)

    def custom_evaluation(self, i, net_out, downstream_preds):
        """
        Compute MAE on test sets
        @param i: the iteration number
        @param h: the current hypothesis
        @param net_out: neural network output
        @param downstream_preds: downstream task predictions
        """
        return
        
def clone_with_transform(base_ds: MNISTSum2Dataset, data_dir, train_flag: bool, transform, sym=False):
        if not sym:
            ds = MNISTSum2Dataset(
                data_dir, train=train_flag, download=True, transform=transform, seed = base_ds.seed
            )
        ds.index_map = list(base_ds.index_map)
        return ds

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


    train_ds_ns = clone_with_transform(train_dataset, data_dir,True, mnist_img_transform)

    test_ds_ns= clone_with_transform(test_dataset, data_dir,False, mnist_img_transform)


    train_loader_ns = torch.utils.data.DataLoader(
        train_ds_ns,
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_train,
        shuffle=True
    )
    test_loader_ns = torch.utils.data.DataLoader(
        test_ds_ns,
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=False
    )
    return train_loader_ns, test_loader_ns

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

    def forward(self, x, **kwargs):
        neural = kwargs.get('neural', False)
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
        if neural is True:
            return x
        else:
            return F.softmax(x,dim=1)

def replace_bn_with_gn(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(num_groups, child.num_features)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)    

class MNISTNet_b0_sym(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b0_sym, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        replace_bn_with_gn(self.model)
        self.modelname = "b0"
        self.classifier = nn.Linear(self.model.num_features, N)
        

    def forward(self, x,**kwargs):   
        neural = kwargs.get('neural', False)
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        if neural is True:
            return x
        else:
            return F.softmax(x,dim=1)

class MNISTNet_b3_sym(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b3_sym, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        replace_bn_with_gn(self.model)
        self.modelname = "b3"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x, **kwargs):
        neural = kwargs.get('neural', False)
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        if neural is True:
            return x
        else:
            return F.softmax(x,dim=1)

class MNISTNet_b0(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b0, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.modelname = "b0"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x, **kwargs):
        neural = kwargs.get('neural',False)
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        if neural is True:
            return x
        else:
            return F.softmax(x,dim=1)

class MNISTNet_b3(MNISTNet):
    def __init__(self, N=10):
        super(MNISTNet_b3, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.modelname = "b3"
        self.classifier = nn.Linear(self.model.num_features, N)

    def forward(self, x, **kwargs):
        neural = kwargs.get('neural', False)
        x = self.model(x) #estrae feature
        x = self.classifier(x) #fa classificazione
        if neural is True:
            return x
        else:
            return F.softmax(x,dim=1)

class MNISTSum2Net(nn.Module):
    def __init__(self, model):
        super(MNISTSum2Net, self).__init__()
        self.mnist_net = model
        self.feature_dim = 10

        combined_dim = self.feature_dim * 2
        self.sum_classifier = nn.Linear(combined_dim, 19)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        (a_imgs, b_imgs) = x

        # First recognize the two digits
        a_feat = self.mnist_net(a_imgs, neural=True)
        b_feat = self.mnist_net(b_imgs, neural=True)

        a_pred = (F.softmax(a_feat,dim=1)).argmax(dim=1)
        b_pred = (F.softmax(b_feat,dim=1)).argmax(dim=1)

        combined_feature = torch.cat((a_feat,b_feat), dim=1)

        sumpred = F.softmax(self.sum_classifier(combined_feature), dim=1)
        return sumpred, a_pred, b_pred

def conv_sum(d1, d2):  
    B = d1.size(0)
    x = d1.unsqueeze(0)                     
    w = d2.unsqueeze(1).flip(2)             
    y = nn.functional.conv1d(x, w, groups=B, padding=9)  
    p_sum = y.squeeze(0)[:, :19]             
    p_sum = p_sum / (p_sum.sum(dim=1, keepdim=True) + 1e-12)
    return p_sum

class Trainer_Sym():
    def __init__(self, transform, test_loader_ns, data_dir, model_dir, learning_rate, model : MNISTNet, device, cfg):
        self.model_dir = model_dir
        self.network = model
        self.data_dir = data_dir
        self.device = device
        self.network.to(self.device)
        which_task = CustomArgument(a_name='task_type', a_type=str, a_default='sum', a_help='sum')
        which_image = CustomArgument(a_name='image_type', a_type=str, a_default='mnist', a_help='mnist')
        NSIL._downstream_task_eval = batched_downstream_task_eval
        self.nsil = NSILRun(custom_args=[which_task, which_image])
        self.net_name = 'digit'
        self.nsil.args.num_iterations = cfg.num_iterations
        self.nsil.args.net_num_epochs = cfg.epochs
        self.nsil.args.net_batch_size = cfg.batch_size
        self.nsil.args.seed = cfg.seed
        self.nsil.args.pct = cfg.pct
        set_random_seeds(cfg.seed)
        self.nsil.args.lr = learning_rate
        self.nsil.args.num_workers = cfg.num_workers
        self.train_data = MNISTSum2NSIL(self.nsil,self.data_dir,transform,cfg.seed,cfg.num_workers,cfg.batch_size)
        self.batchsize = cfg.batch_size
        image_to_network_map = {
            'i1': self.net_name,
            'i2': self.net_name,
        }
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        net_confs = {self.net_name: NSILNetworkConfiguration(name=self.net_name, net=self.network, num_out=10, optim=self.optimizer)}
        self.test_loader_ns = test_loader_ns 
        self.loss = nn.NLLLoss()
        self.task = ArithmeticTask(self.train_data, self.nsil, net_confs, image_to_network_map, ilp_config=self.nsil.args.ilp_config,digit_vals=list(range(0, 10)))

    def train(self): 
        self.nsil.nsil = NSIL(self.nsil.args, self.nsil.logger, self.task)
        self.nsil.nsil.train()
        torch.save(self.network.state_dict(), os.path.join(self.model_dir, f"{self.network.modelname}_nesy.pt"))
        acc_train,sdacc=self.gen_results()
        return {
            "loss": None,
            "accuracy" : acc_train,
            "single-digit accuracy": (sdacc,sdacc)
        }
        
    def gen_results(self):
        run_dir = "./results/runs"
        runs = os.scandir(run_dir)
        int_runs = [int(r.name.split('_')[1]) for r in runs if r.is_dir()]
        if len(int_runs) > 0:
                int_runs.sort()
                run_name = 'run_{0}'.format(int_runs[-1])
        log_dir = os.path.join(run_dir, run_name)
        with open(os.path.join(log_dir,"test_log.json"),"r") as f:
            training_results = json.load(f)
        acc_train = []
        sdacc = []
        for it in training_results:
            if it == 0:
                continue
            acc_train.append(training_results[it]["end_to_end_acc"])
            sdacc.append(training_results[it]["network_accuracy"][self.net_name])
        return acc_train, max(sdacc)

    def test(self, dataloader = None, desc="Test Loop"):
        dataloader = self.test_loader_ns if dataloader is None else dataloader
        self.network.eval()
        y_true, y_pred = [], []
        a_true, a_pred = [], []
        b_true, b_pred = [], []
        running_loss = 0.0
        self.network.to(self.device)
        with torch.no_grad():
            for ((img1,img2), (d1,d2)) in tqdm(dataloader, total=len(dataloader), desc=desc):
                img1,img2 = img1.to(self.device), img2.to(self.device)
                target = d1 + d2
                target = target.to(self.device)
                dprob1 = self.network(img1)
                dprob2 = self.network(img2)
                output = conv_sum(dprob1,dprob2)
                running_loss += self.loss(torch.log(output + 1e-9), target.long()).item() * target.size(0)

                preds = output.argmax(dim=1)
                dpred1 = dprob1.argmax(dim=1)
                dpred2 = dprob2.argmax(dim=1)

                y_true.extend(target.cpu().tolist())
                y_pred.extend(preds.detach().cpu().tolist())

                a_true.extend(d1.cpu().tolist())
                a_pred.extend(dpred1.detach().cpu().tolist())

                b_true.extend(d2.cpu().tolist())
                b_pred.extend(dpred2.detach().cpu().tolist())
            

        test_loss = running_loss / len(dataloader.dataset)
        acc = accuracy_score(y_true, y_pred)
        d1acc = accuracy_score(a_true,a_pred)
        d2acc = accuracy_score(b_true,b_pred)

        if dataloader.dataset.dataset_name == "test":
            print(f"- {self.network.modelname} nesy model - Test loss: {test_loss} with total accuracy: {acc*100}% \n digit1 accuracy: {d1acc} and digit2 accuracy: {d2acc}")
        
        return {
            "loss": test_loss,
            "accuracy" : acc,
            "single-digit accuracy": (d1acc, d2acc)
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
