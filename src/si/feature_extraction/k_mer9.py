import itertools
import numpy as np
from si.data.dataset import Dataset

class KMer9:
    """
    A sequence descriptor that returns the k-mer composition of the sequence.
    Parameters
    ----------
    k : int
        The k-mer length.
    alphabet: str
        The biological sequence alphabet
    Attributes
    ----------
    k_mers : list of str
        The k-mers.
    """
    def __init__(self, k: int = 2, alphabet: str = "ACTG"):
        """
        Parameters
        ----------
        k : int
            The k-mer length.
        alphabet: str
        The biological sequence alphabet
        """
        # parameters
        self.k = k
        self.alphabet = alphabet

        # attributes
        self.k_mers = None

    def fit(self, dataset: Dataset) -> 'KMer':
        """
        Fits the descriptor to the dataset.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the descriptor to.
        Returns
        -------
        KMer
            The fitted descriptor.
        """
        # generate the k-mers
        #self.k_mers = [''.join(k_mer) for k_mer in itertools.product('ACTG', repeat=self.k)]

        self.k_mers= []
        for k_mer in itertools.product(self.alphabet, repeat=self.k):
            self.k_mers.append(''.join(k_mer))
        return self

    def _get_sequence_k_mer_composition(self, sequence: str) -> np.ndarray:
        """
        Calculates the k-mer composition of the sequence.
        Parameters
        ----------
        sequence : str
            The sequence to calculate the k-mer composition for.
        Returns
        -------
        list of float
            The k-mer composition of the sequence.
        """
        # calculate the k-mer composition
        counts = {k_mer: 0 for k_mer in self.k_mers}

        for i in range(len(sequence) - self.k + 1):
            k_mer = sequence[i:i + self.k]
            counts[k_mer] += 1

        # normalize the counts
        return np.array([counts[k_mer] / len(sequence) for k_mer in self.k_mers])

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset.
        Parameters
        ----------
        dataset : Dataset
            The dataset to transform.
        Returns
        -------
        Dataset
            The transformed dataset.
        """
        # calculate the k-mer composition
        #sequences_k_mer_composition = [self._get_sequence_k_mer_composition(sequence)
         #                              for sequence in dataset.X[:, 0]]

        sequences_k_mer_composition=[]
        for sequence in dataset.X[:, 0]:
            sequences_k_mer_composition.append(self._get_sequence_k_mer_composition(sequence))

        sequences_k_mer_composition = np.array(sequences_k_mer_composition)

        # create a new dataset
        return Dataset(X=sequences_k_mer_composition, y=dataset.y, features=self.k_mers, label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the descriptor to the dataset and transforms the dataset.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the descriptor to and transform.
        Returns
        -------
        Dataset
            The transformed dataset.
        """
        return self.fit(dataset).transform(dataset)


if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset_ = Dataset(X=np.array([['ACTGTTTAGChujiGGA', 'ACTGTToiujhbTAGCGGA']]),
                       y=np.array([1, 0]),
                       features=['sequence'],
                       label='label')

    k_mer_ = KMer9(k=2, alphabet="ACTGhujiob")
    dataset_1 = k_mer_.fit_transform(dataset_)

    #print(dataset_.X)
    #print(dataset_.features)
    print(str(len(dataset_1.features)))
    k_mer_ = KMer9(k=2, alphabet="ACTGTToiujhbTAGCGGAACTGTTTAGChujiGGA")
    dataset_2 = k_mer_.fit_transform(dataset_)
    #print(dataset_.X)
    #print(dataset_.features)
    print(str(len(dataset_2.features)))