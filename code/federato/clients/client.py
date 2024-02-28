import os
import argparse
import warnings
from collections import OrderedDict
from typing import Dict
import pandas as pd
import time

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common import Metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Pad
from tqdm import tqdm
from noniidpart import NonIidPartitioner
from nets import NetMNIST, NetFMNIST, NetCifar10

# #############################################################################
# 1. Pipeline regolare di PyTorch: nn.Module, train, test e DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cpu")

def model_picker(dataset):
    if dataset == "cifar10":
        model = NetCifar10()
    elif dataset == "fashion_mnist":
        model = NetFMNIST()
    else:
        model = NetMNIST()
    return model

def train(net, dtype, trainloader, epochs):
    """Allena il modello sul train set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    #Sono create delle liste vuote per fornire indietro i valori ottenuti durante il processo
    train_losses = []
    train_accuracies = []
    epoch_times = []
    net.train()
    loss = 0.0
    for _ in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        start_time = time.time()
        for batch in tqdm(trainloader, "Training"):
            #Viene effettuato un controllo per ottenere le immagini in base al dataset utilizzato
            if dtype == "cifar10":
            	images = batch["img"].to(DEVICE)
            else:
            	images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        end_time = time.time()
        epoch_times.append(float(end_time - start_time))
        train_losses.append(float(epoch_loss / len(trainloader.dataset)))
        train_accuracies.append(float(correct / total))
    return train_losses, train_accuracies, epoch_times
 


def test(net, dtype, testloader):
    """Valida il modello sul test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    true_labels = []
    predicted_labels = []
    net.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch in tqdm(testloader, "Testing"):
            if dtype == "cifar10":
            	images = batch["img"].to(DEVICE)
            else:
            	images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return float(loss), float(accuracy), time.time() - start_time, true_labels, predicted_labels

def transform_choice(choice):
    """Funzione adoperata per selezionare i cambi da effettuare sulle immagini in base al dataset adoperato"""
    if choice == "cifar10":
        transform = Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )
    elif choice == "fashion_mnist":
        transform = Compose(
        [Pad(2, fill=0), ToTensor(), Normalize((0.5,), (0.5,))]
        )
    else:
        transform = Compose(
        [Pad(2, fill=0), ToTensor(), Normalize((0.1307,), (0.3081,))]
        )
    return transform

def load_data(node_id, clients, dtype, ptype):
    """Funzione adoperata per caricare la partizione del dataset selezionato."""
    part_dim = 10
    batch = 64
    if ptype == "noniid":
        """La dimensione della partizione è data dal partitioner adoperato"""
        fds = FederatedDataset(dataset=dtype, partitioners={"train": NonIidPartitioner("label", clients, part_dim), "test": NonIidPartitioner("label", clients, part_dim)})
    else:
        """La dimensione della partizione è data dal valore della partizione generale per il numero dei client"""
        fds = FederatedDataset(dataset=dtype, partitioners={"train": clients * part_dim, "test": clients * part_dim})
    
    partition_train = fds.load_partition(node_id, 'train')
    partition_test = fds.load_partition(node_id, 'test')
    print(f"Dimensione del trainset del client: {len(partition_train)}")
    print(f"Dimensione del testset del client: {len(partition_test)}")
    
    #Sono definiti i cambi da applicare all'immagine
    pytorch_transforms = transform_choice(dtype)

    def apply_transforms_cif(batch):
    	"""Applica le transformazioni al dataset partizionato, esclusivamente per cifar10"""
    	batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    	return batch
    
    def apply_transforms(batch):
    	"""Applica le transformazioni al dataset partizionato"""
    	batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    	return batch

    if dtype == "cifar10":
    	partition_train = partition_train.with_transform(apply_transforms_cif)
    	partition_test = partition_test.with_transform(apply_transforms_cif)
    else:
    	partition_train = partition_train.with_transform(apply_transforms)
    	partition_test = partition_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train, batch_size=batch, shuffle=True)
    testloader = DataLoader(partition_test, batch_size=batch)
    print(len(trainloader), len(testloader))
    return trainloader, testloader


# #############################################################################
# 2. Federazione della pipeline con Flower
# #############################################################################

def argparser_func():
    parser = argparse.ArgumentParser(description="Impostazioni Flower client")
    #Impostazioni disponibili da linea di comando
    parser.add_argument(
        "--dataset",
        choices=['mnist', 'cifar10', 'fashion_mnist'],
        type=str,
        default='mnist',
        help="Dataset utilizzato per il training (default: mnist)",
    )
    parser.add_argument(
        "--partition",
        choices=['iid', 'noniid'],
        type=str,
        default='iid',
        help="Tipo di partitioning usato per il dataset (default: iid)",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="192.168.128.230:8080",
        help=f"Indirizzo server (deafault '192.168.128.230:8080')",
    )
    parser.add_argument(
        "--node-id",
        choices=[0, 1, 2],
        required=True,
        type=int,
        help="Partizione del dataset diviso in 3 partizioni non iid create artificialmente.",
    )
    return parser.parse_args()

def save_to_csv(loss_values, accuracy_values, time_values, filename, flag_file):
    #E' creato un dataframe dalla lista
    if flag_file:
    	df = pd.DataFrame({
    		'Loss': loss_values,
    		'Accuratezza': accuracy_values,
    		'MTPE': time_values
        }, index=[0])
    else:
    	df = pd.DataFrame({
    		'Loss': loss_values,
    		'Accuratezza': accuracy_values,
    		'MTPE': time_values
    	})
    
    #Il dataframe è salvato in un file csv
    df.to_csv(filename, mode='a', index=False)

#E' definito il client di Flower
class FlowerClient(fl.client.NumPyClient):
    
    def __init__(self, node_id, dtype, net, trainloader, testloader, parent_directory, ptype):
        self.node_id = node_id
        self.dtype = dtype
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.parent_directory = parent_directory
        self.ptype = ptype
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        epoch: int = config["local_epochs"]
        
        train_losses, train_accuracies, train_times = train(self.net, self.dtype, self.trainloader, epochs=epoch)
        save_to_csv(train_losses, train_accuracies, train_times, self.parent_directory + '/' + f'{self.node_id}_epoch_train_{self.ptype}.csv', False)
        loss_dict = {f"loss_{i}": loss for i, loss in enumerate(train_losses)}
        accuracy_dict = {f"accuracy_{i}": accuracy for i, accuracy in enumerate(train_accuracies)}
        metrics = {**loss_dict, **accuracy_dict}
        return self.get_parameters(config={}), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        loss, accuracy, time, true_labels, predicted_labels = test(self.net, self.dtype, self.testloader)
        save_to_csv(loss, accuracy, time, self.parent_directory + '/' + f'{self.node_id}_epoch_test_{self.ptype}.csv', True)
        tr_lab_dic = {f"true_{i}": true for i, true in enumerate(true_labels)}
        pr_lab_dic = {f"predicted_{i}": predicted for i, predicted in enumerate(predicted_labels)}
        res = {**tr_lab_dic, **pr_lab_dic}
        acc_loss = {"accuracy": accuracy, "loss": loss}
        metrics={**acc_loss,**res}
        return loss, len(self.testloader.dataset), metrics

def main():
    args = argparser_func()
    directory_name = args.dataset + "_" + args.partition
    parent_directory = os.path.expanduser(f"~/Scaricati/proj_client/")
    #Viene combinato il path
    path = os.path.join(parent_directory, directory_name)
    #E' creata la nuova directory
    os.makedirs(path, exist_ok=True)
    print(f"Directory '{directory_name}' creata")
    net = model_picker(args.dataset).to(DEVICE)
    num_clients = 3
    #Carica il modello ed i dataset (ad esempio una rete CNN con MNIST)
    trainloader, testloader = load_data(args.node_id, num_clients, args.dataset, args.partition)
    #Avvia il client di Flower
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=FlowerClient(args.node_id, args.dataset, net, trainloader, testloader, parent_directory + directory_name, args.partition),
    )

if __name__ == "__main__":
    main()
