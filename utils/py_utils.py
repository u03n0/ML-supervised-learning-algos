from random import shuffle
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
from numba import jit



def build_dataset(path: Path)-> list[dict[str, str]]:
    """ Reads csv file and builds dataset as a list of dicts
    """
    l: list = []
    with open(path, 'r') as file:
       for line in file:
           line = line.replace('\n', '')
           key, *value = line.split(",")
           value_str = " ".join(value)
           l.append({key: value_str})
    return l[1:-1]

def clean_dataset(list_: list[dict[str, str]], stoplist: list = [])-> list[dict[str, str]]:
    """ Cleans a dataset by removing non-letters from the text, lowering,
    and filtering out stopwords.
    """
    result: list = []
    for dict_ in list_:
        for key, value in dict_.items():
            cleaned_text = remove_non_alpha(value, stoplist)
        result.append({key: cleaned_text})
    return result

def remove_non_alpha(text: str, stoplist: list = [])-> str:
    """ Removes all non-letters from a text (string)
    and checks if not in stoplist.
    """
    word_list: list = []
    for word in text.split():
        if word.isalpha and word.lower() not in stoplist:
            word_list.append(word.lower())
    return " ".join(word_list)


def tf(term: str, document: str)-> float:
    """ Term Frequency calculator.
    Relative frequency of term t within a document d 
    """
    text = document.split()
    n: int = len(text)
    freq: int = text.count(term)
    return freq / n if n else 0


def idf(term: str, corpus: list[dict[str, list[str]]])-> float:
    """ Inverse Document Frequency. Log of number of 
    documents (N) in corpus (D) over 1 + the number of 
    documents (d) within the corpus (D) that the term (t)
    appears in. 
    """
    n: int = len(corpus)
    docs_with_term_count = sum(1 for doc in corpus if any(term in value.split() for value in doc.values()))
    if docs_with_term_count == 0:
        return 0
    return math.log((n + 1) / (docs_with_term_count + 1)) + 1


def get_vocab(corpus: list[dict[str, str]])-> set[str]:
    """
    Returns a set of unique words (vocabulary) in the entire corpus.
    """
    vocab: set = set()
    for doc in corpus:
        for value in doc.values():
            for term in value.split():
                vocab.add(term)
    return vocab


def get_tf_idf(corpus: list[dict[str, str]])-> list[dict[str, list[float]]]:
    """ Creates a list of dictionaries in which the key is a label (str) and 
    the value is a vector (list of tf-idf) that represents a text.

    """     
    vocab: set = get_vocab(corpus) # Get a set of all terms in corpus
    vocab_len: int = len(vocab)
    term_idfs: dict[str, float] = {term: idf(term, corpus) for term in vocab} # Get idf of all terms in vocab
    results: list = [] # List to store results 

    for document in corpus:
        term_freq = defaultdict(int) # Dict to hold tf of each term in a document.
        vector = [0] * vocab_len # Vector of 0s x length of vocab in corpus
        label = list(document.keys())[0] # The label is the key of each document (dict)

        for text in document.values():
            for term  in text.split():
                term_freq[term] = tf(term, text)

        for i, term in enumerate(vocab):
            vector[i] = term_freq.get(term, 0) * term_idfs[term] 

        results.append({label: vector})

    return results


def train_test_split(dataset: list[dict[str, list[str]]], ratio: float)-> tuple[list[dict[str, list[str]]], list[dict[str, list[str]]]]:
    """ Simple train test split of a dataset,
    that is a namedtuple.
    """
    shuffle(dataset)
    split_ratio = ratio
    split_index = int(split_ratio * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data


@jit(nopython=True)
def cosine_similarity(a: list[float], b: list[float])-> float:
    """ The dot product of two vectors (a, b) divided
    by the product of the magnitudes of the vectors.
    cosine similarity = A * B / ||A|| * ||B||
    """
    a = np.array(a)
    b = np.array(b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    return dot_product / (norm_a * norm_b)
