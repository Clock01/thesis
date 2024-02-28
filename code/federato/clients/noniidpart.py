from typing import Dict
import datasets
from flwr_datasets.partitioner.natural_id_partitioner import NaturalIdPartitioner
import numpy as np

class NonIidPartitioner(NaturalIdPartitioner):
    """Partizionatore per un set di dati in cui la partizione finale risulta in diversi set di dati
    non sovrapposti per un dato numero di gruppi/nodi"""

    def __init__(self, partition_by, num_nodes, part_dim) -> None:
        super().__init__(partition_by)
        self._num_nodes = num_nodes
        self._part_dim = part_dim

    def _create_int_node_id_to_natural_id(self) -> None:
        """Crea una mappatura da indici int a id univoci di client o gruppi dal set di dati.
        Gli id naturali provengono dalla colonna specificata in `partition_by`.
        """
        
        unique_natural_ids = self.dataset.unique(self._partition_by)

        #Suddivide le etichette tra il numero di nodi/gruppi
        split_natural_ids = np.array_split(unique_natural_ids, self._num_nodes)
        print(split_natural_ids)

        self._node_id_to_natural_id = dict(
            zip(range(self._num_nodes), split_natural_ids)
        )

    def load_partition(self, node_id: int) -> datasets.Dataset:
        """Carica una singola partizione corrispondente a un singolo `node_id`.
        La scelta della partizione si basa sul parametro node_id, 
        e sulla mappatura calcolata nella funzione 
        _create_int_node_id_to_natural_id()
        Parametri
        ----------
        node_id : int
            l'indice che corrisponde alla partizione richiesta
        Ritorna
        -------
        dataset_partition : Dataset
            partizione di un singolo set di dati
        """
        if len(self._node_id_to_natural_id) == 0:
            self._create_int_node_id_to_natural_id()

        data = self.dataset.filter(
            lambda row: row[self._partition_by] in self._node_id_to_natural_id[node_id]
        )
        print(len(data))
        #E' fornito indietro lo shard al client con node_id utilizzato alla chiamata della funzione
        return data.shard(
            num_shards=self._part_dim, index=node_id, contiguous=True
        )



    @property
    def node_id_to_natural_id(self) -> Dict[int, str]:
        """Id del nodo al corrispondente id naturale presente.
        Gli id naturali sono i valori univoci della colonna `partition_by` nel set di dati..
        """
        return self._node_id_to_natural_id

    # pylint: disable=R0201
    @node_id_to_natural_id.setter
    def node_id_to_natural_id(self, value: Dict[int, str]) -> None:
        raise AttributeError(
            "L'impostazione del dizionario node_id_to_natural_id non è consentita."
        )
