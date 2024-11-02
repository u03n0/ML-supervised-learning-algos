import csv
from random import shuffle
from collections import namedtuple
from typing import List, Dict, Tuple
from math import log



def build_dataset(path: str)-> Dict[str, str]:
    """ Reads csv file and builds dataset as a list of dicts
    """
    with open(path, 'r') as file:
        return [dict(*row) for row in csv.reader(file) if all(row)][1:]

    
def tf(term: str, document: str):
    """ Term Frequency calculator.
    Relative frequency of term t within a document d 
    """
    n = len(document.split())
    freq = 0
    for word in document.split():
        if word == term:
            freq += 1

    if freq:
        return (freq / n) + 1
    else:
        return 1 / n 


def idf(term: str, corpus: List[namedtuple]):
    """ Inverse Document Frequency. Log of number of 
    documents (N) in corpus (D) over 1 + the number of 
    documents (d) within the corpus (D) that the term (t)
    appears in. 
    """
    n = len(corpus)
    doc_count = sum(1 for doc in corpus if term in doc.text)
    return log(n / (1 + doc_count)) + 1

def tf_idf(term: str, document: str, corpus: List[namedtuple]):
    """ Term Frequency-Inverse Document Frequency.

    """

    return tf(term, document) * idf(term, corpus)

def train_test_split(dataset: List[namedtuple], ratio: float)-> Tuple[List[namedtuple]]:
    """ Simple train test split of a dataset,
    that is a namedtuple.
    """
    shuffle(dataset)
    split_ratio = ratio
    split_index = int(split_ratio * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data


