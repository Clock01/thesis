import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from nets import NetMNIST, NetFMNIST, NetCifar10

def model_picker(dataset):
    if dataset == "cifar10":
        model = NetCifar10()
    elif dataset == "fashionmnist":
        model = NetFMNIST()
    else:
        model = NetMNIST()
    return model


def dataset_partitioning(data):
    #Viene generata una lista di indici per l'intero dataset
    dataset_size = len(data)
    dataset_indices = list(range(dataset_size))

    #Essi sono in seguito mescolati
    np.random.shuffle(dataset_indices)

    #Viene stabilita la dimensione del subsample ed in seguito sono selezionati il numero di indici
    sample_size = int(dataset_size / 10)
    subsample_indices = dataset_indices[0:sample_size]
     
    #Viene creato un oggetto Subset usando gli indici selezionati
    return Subset(data, subsample_indices)

def mnist(pytorch_transforms):
	"""Funzione per il reperimento del train e test set di MNIST
    Se non presente viene scaricato.
    """
	train_data = datasets.MNIST(
		root='data',
		train=True,
		download=True,
		transform=pytorch_transforms
	)
	test_data = datasets.MNIST(
		root='data',
		train=False,
		download=True,
		transform=pytorch_transforms
	)
	return train_data, test_data

def cifar10(pytorch_transforms):
	"""Funzione per il reperimento del train e test set di Cifar10
    Se non presente viene scaricato.
    """
	train_data = datasets.CIFAR10(
		root='data',
		train=True,
		download=True,
		transform=pytorch_transforms
	)
	test_data = datasets.CIFAR10(
		root='data',
		train=False,
		download=True,
		transform=pytorch_transforms
	)
	return train_data, test_data


def fashmnist(pytorch_transforms):
	"""Funzione per il reperimento del train e test set di Fashion MNIST
    Se non presente viene scaricato.
    """
	train_data = datasets.FashionMNIST(
		root='data',
		train=True,
		download=True,
		transform=pytorch_transforms
	)
	test_data = datasets.FashionMNIST(
		root='data',
		train=False,
		download=True,
		transform=pytorch_transforms
	)
	return train_data, test_data


def transform_choice(choice):
    if choice == "cifar10":
        transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )
    elif choice == "fashionmnist":
        transform = transforms.Compose(
        [transforms.Pad(2, fill=0), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    else:
        transform = transforms.Compose(
        [transforms.Pad(2, fill=0), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    return transform


def load_data(batch, choice):
    pytorch_transforms = transform_choice(choice)
    #Carica il training dataset e se assente procede a scaricarlo
    if choice == "cifar10":
    	train_data, test_data = cifar10(pytorch_transforms)
    elif choice == "fashionmnist":
        train_data, test_data = fashmnist(pytorch_transforms)
    else:
        train_data, test_data = mnist(pytorch_transforms)
    sampler = dataset_partitioning(train_data)
    test_sampler = dataset_partitioning(test_data)
    #Sono creati i loader per i dati
    train_loader = DataLoader(sampler, batch_size=batch, shuffle=True)
    print(len(train_loader.dataset))
    test_loader = DataLoader(test_sampler, batch_size=batch)
    print(len(test_loader.dataset))
    return train_loader, test_loader    

# Viene creata la funzione per ottenere la perdità e l'accuratezza
def train(model, device, train_loader, criterion, optimizer, epoch):
    correct, total, running_loss = 0, 0, 0.0
    model.train()
    start_time = time.time()
    for i, (data, target) in enumerate(train_loader, 0):
        inputs, labels = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if i < 1 or i % 10000 == 0: #47
        	print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
    return float(running_loss / (len(train_loader.dataset))), float(100 * correct / total), time.time() - start_time


def test(model, device, test_loader, criterion):
    # Viene effettuato il calcolo della perdita e l'accuratezza del test dataset
    model.eval()
    true_labels = []
    predicted_labels = []
    total, correct, loss = 0, 0, 0.0
    with torch.no_grad():
        start_time = time.time()
        for data, target in test_loader:
            images, labels = data.to(device), target.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= (len(test_loader.dataset))
    print('Accuratezza della rete su 1000 immagini di test: %d %%' % (100 * correct / total))
    return float(loss), float(100 * correct / total), time.time() - start_time, true_labels, predicted_labels


def save_to_csv(loss_values, accuracy_values, time_values, filename):
    # E' creato un dataframe dalla lista
    	
    df = pd.DataFrame({
        'Loss': loss_values,
        'Accuratezza': accuracy_values,
        'MTPE': time_values
    })
    
    # Il dataframe è salvato in un file csv
    df.to_csv(filename, mode='a', index=False)


def plot_results(loss, accuracies, times, n_img, epoch_num, directory):

   #Grafico della perdita in relazione alle epoche
   plt.figure()
   plt.grid()
   plt.plot(range(0, epoch_num), loss)
   plt.title(f'{n_img} Loss per epoca')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.savefig(directory + '/' + f'{n_img}_loss.png') #Salva il grafico nel file

   #Grafico dell'accuretezza in relazione alle epoche
   plt.figure()
   plt.grid()
   plt.plot(range(0, epoch_num), accuracies)
   plt.title(f'{n_img} Accuratezza per epoca')
   plt.xlabel('Epoch')
   plt.ylabel('Accuratezza (%)')
   plt.savefig(directory + '/' + f'{n_img}_accuracy.png') #Salva il grafico nel file
   
   save_to_csv(loss, accuracies, times, directory + '/' + f'epoch_{n_img}.csv')


def lab_choice(dtype):
    if dtype == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dtype == "fashion_mnist":
        return ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def print_conf_mat(true_labels, predicted_labels, directory, labels):
    #Viene computata la matrice di confusione
    cm = confusion_matrix(true_labels, predicted_labels)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    #Viene graficata la mtrice
    plt.figure(figsize=(14, 14))
    plt.matshow(cm, cmap='Blues')
    plt.title('Matrice di confusione')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xlabel('Etichette predette')
    plt.xticks(tick_marks, labels, rotation=40)
    plt.ylabel('Etichette vere')
    plt.yticks(tick_marks, labels, rotation=70)
    
    
    plt.savefig(f'{directory}/confusion_matrix.png')

def mk_dir(epochs, dataset):
    directory_name = str(epochs) + "_plots_" +  dataset
    #E' definità la directory parente
    parent_directory = os.path.expanduser("~/Scaricati/proj_central/")
    #Viene combinato il path
    path = os.path.join(parent_directory, directory_name)
    #E' creata la nuova directory
    os.makedirs(path, exist_ok=True)
    print(f"Directory '{directory_name}' creata")
    return parent_directory + directory_name

def argparser_func():
    parser = argparse.ArgumentParser(description="Impostazioni learning centralizzato")
    #Impostazioni disponibili da linea di comando
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Numero di epoche per il dispositivo (default: 1)",
        )
        
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Dimensione dei batch (default: 64)",
        )
    
    parser.add_argument(
        "--dataset",
        choices=['mnist', 'cifar10', 'fashionmnist'],
        type=str,
        default='mnist',
        help="Dataset utilizzato per il training (default: mnist)",
        )
    
    return parser.parse_args()

def main():
    #Liste adoperate per la stampa dei grafici e la conservazione dei risultati nei file csv
    test_losses = []
    test_accuracies = []
    test_times = []
    train_losses = []
    train_accuracies = []
    train_times = []
    true_labels = []
    predicted_labels = []
    
    args = argparser_func()
    directory_name = mk_dir(args.epochs, args.dataset)
    labels = lab_choice(args.dataset)
    device = torch.device("cpu")
    #E' inizializzata la rete ed è definito l'ottimizzatore
    model = model_picker(args.dataset)
    #Si usa come ottimizzatore la Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum = 0.9, weight_decay=1e-4)
    
    #E' defintita la loss function
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = load_data(args.batch_size, args.dataset)
    #E' allenato il modello e sono stampati i grafici dei risultati
    for epoch in range(args.epochs):
        train_loss, train_accuracy, train_time = train(model, device, train_loader, criterion, optimizer, epoch)
        test_loss, test_accuracy, test_time, true_labels, predicted_labels = test(model, device, test_loader, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_times.append(train_time)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_times.append(test_time)
    plot_results(train_losses, train_accuracies, train_times, 'Train', args.epochs, directory_name)
    plot_results(test_losses, test_accuracies, test_times, 'Test', args.epochs, directory_name)
    print_conf_mat(true_labels, predicted_labels, directory_name, labels)


if __name__ == "__main__":
    main()
