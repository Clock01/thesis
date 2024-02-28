"""Flower server."""
import os
import argparse
import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd
import numpy as np
import flwr as fl
from flwr.common import Metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""Classe contenente le funzioni adoperate per l'aggregazione dei valori ottenuti dai client"""
class ServerFlwr:
    def __init__(self, rounds, epochs, dataset, partition):
        self.rounds = rounds
        self.epochs = epochs
        self.dataset = dataset
        self.partition = partition
        self.parent_directory = mk_dir(self.rounds, self.epochs, self.dataset, self.partition)
        self.labels = lab_choice(self.dataset)
        
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []
        
    #E' definita la funzione di aggragazione dei valori di test
    def weighted_average_test(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
        #L'accuratezza è ottenuta moltiplicando l'accuratezza di ogni client per gli il numero di esempi utilizzati
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        all_true_labels = []
        all_predicted_labels = []
        for num_examples, m in metrics:
            sorted_dict = dict(sorted(m.items()))
            true_labels = [value for key, value in sorted_dict.items() if "true_" in key]
            predicted_labels = [value for key, value in sorted_dict.items() if "predicted_" in key]
            all_true_labels.extend(true_labels)
            all_predicted_labels.extend(predicted_labels)
        #Stesso approccio è utilizzato per la perdità
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        print_conf_mat(all_true_labels, all_predicted_labels, self.parent_directory, self.labels)
        #I valori sono salvati all'interno di vettori per la stampa dei grafici
        self.test_loss.append(sum(losses) / sum(examples))
        self.test_accuracy.append(100 * sum(accuracies) / sum(examples))
        #I valori aggregati sono restituiti come metrica
        return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}
    
    #E' definita la funzione di aggragazione dei valori di train
    def weighted_average_train(self, metrics: List[Tuple[int, Metrics]]) -> Tuple[List[float], List[float]]:
        total_loss = 0
        total_accuracy = 0
        total_examples = 0
        
        loss = [0] * self.epochs
        accuracy = [0] * self.epochs
        for num_examples, m in metrics:
            sorted_dict = dict(sorted(m.items()))
            # Vengono estratti i valori relativi alle perdite ed alla accuratezza dalle metriche
            epoch_losses = [value for key, value in sorted_dict.items() if "loss_" in key]
            epoch_accuracies = [value for key, value in sorted_dict.items() if "accuracy_" in key]
            # I valori ottenuti durante le epoche sono inseriti nelle apposite liste
            for i in range(self.epochs):
            	loss[i] += (epoch_losses[i] / 3)
            	accuracy[i] += ((100 * epoch_accuracies[i]) / 3)
            # Sono calcolate le somme dei pesi delle perdite e della accuratezza
            total_loss += sum([num_examples * loss for loss in epoch_losses])
            total_accuracy += sum([num_examples * accuracy for accuracy in epoch_accuracies])
            total_examples += num_examples
        self.train_loss.extend(loss)
        self.train_accuracy.extend(accuracy)
        #I valori aggregati sono restituiti come metrica
        return {"accuracy": total_accuracy / total_examples, "loss": total_loss / total_examples}
    
    def fit_config(self, server_rounds: int):
        """E' fornito indietro il dizionario con le configurazioni del training.
        """
        config = {
            "local_epochs": self.epochs,
        }
        return config
        
    def strategy_generator(self):
        #Viene fornita indietro la strategia utilizzata
        return fl.server.strategy.FedAvg(
		min_available_clients=3,
		min_fit_clients=3,
		min_evaluate_clients=3,
		fit_metrics_aggregation_fn=self.weighted_average_train,
		evaluate_metrics_aggregation_fn=self.weighted_average_test,
		on_fit_config_fn=self.fit_config)
        
    def plot_graphs(self):
        #Sono richiamate le funzioni per la creazione dei grafici
        plot_results(self.train_loss, self.train_accuracy, self.rounds * self.epochs, 'Train', self.parent_directory)
        plot_results(self.test_loss, self.test_accuracy, self.rounds, 'Test', self.parent_directory)
        
        
def argparser_func():
    parser = argparse.ArgumentParser(description="Impostazioni Flower server")
    #Impostazioni disponibili da linea di comando
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help=f"Indirizzo server (deafault '0.0.0.0:8080')",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Numero di round per il federated learning (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Numero di epoche per i client (default: 1)",
    )
    parser.add_argument(
        "--dataset",
        choices=['mnist', 'cifar10', 'fashionmnist'],
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
    
    return parser.parse_args()

def lab_choice(dtype):
    if dtype == 'cifar10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dtype == "fashionmnist":
        return ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def mk_dir(rounds, epochs, dataset, partition):
    directory_name = str(rounds * epochs) + "_plots_" +  dataset + "_" + partition
    #E' definità la directory parente
    parent_directory = os.path.expanduser("~/Scaricati/proj_dat/")
    #Viene combinato il path
    path = os.path.join(parent_directory, directory_name)
    #E' creata la nuova directory
    os.makedirs(path, exist_ok=True)
    print(f"Directory '{directory_name}' creata")
    return parent_directory + directory_name

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

def save_to_csv(loss_values, accuracy_values, filename):
    # E' creato un dataframe dalla lista
    df = pd.DataFrame({
        'Loss': loss_values,
        'Accuratezza': accuracy_values
    })
    
    # Il dataframe è salvato in un file csv
    df.to_csv(filename, index=False)


def plot_results(l, a, epoch_num, n_img, directory):

   # Grafico della perdita in relazione alle epoche
   plt.figure()
   plt.plot(range(0, epoch_num), l)
   plt.grid()
   plt.title('Perdità per epoche complessive')
   plt.xlabel('Epoca')
   plt.ylabel(f'{n_img} Loss')
   plt.savefig(directory + '/' + f'{n_img}_loss.png') # Salva il grafico nel file
   
   # Grafico dell'accuretezza in relazione alle epoche
   plt.figure()
   plt.plot(range(0, epoch_num), a)
   plt.grid()
   plt.title('Accuratezza per round complessivi')
   plt.xlabel('Epoca')
   plt.ylabel(f'{n_img} Accuratezza (%)')
   plt.savefig(directory + '/' + f'{n_img}_accuracy.png') # Salva il grafico nel file
   # E' chiamata la funzione per la creazione dei file csv
   save_to_csv(l, a, directory + '/' + f'{n_img}_results.csv')

def main():
	args = argparser_func()
	sv = ServerFlwr(args.rounds, args.epochs, args.dataset, args.partition)
	#Qui è definita la strategia utilizzata per il learning distribuito
	strategy = sv.strategy_generator()
	#E' fatto partire il server di Flower
	fl.server.start_server(
	    server_address=args.server_address,
	    config=fl.server.ServerConfig(num_rounds=args.rounds),
	    strategy=strategy,
	)
	sv.plot_graphs()
    

if __name__ == "__main__":
    main()
