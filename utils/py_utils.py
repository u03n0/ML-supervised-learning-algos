from random import shuffle
from typing import List, Dict, Tuple
from math import log
from pathlib import Path


def build_dataset(path: str)-> Dict[str, str]:
    """ Reads csv file and builds dataset as a list of dicts
    """
    l = []
    with open(path, 'r') as file:
       for line in file:
           line = line.replace('\n', '')
           key, *value = line.split(",")
           value_str = " ".join(value)
           l.append({key: value_str})
    return l[1:-1]

    
def tf(term: str, document: str):
    """ Term Frequency calculator.
    Relative frequency of term t within a document d 
    """
    n = len(document)
    freq = 0
    for word in document:
        if word == term:
            freq += 1
    if freq:
        return (freq / n) + 1
    else:
        return 1 / n 


def idf(term: str, corpus: List[str]):
    """ Inverse Document Frequency. Log of number of 
    documents (N) in corpus (D) over 1 + the number of 
    documents (d) within the corpus (D) that the term (t)
    appears in. 
    """
    n = len(corpus)
    docs_with_term_count = sum(1 for doc in corpus if term in doc.values())
    return log(n / (1 + docs_with_term_count)) + 1


def tf_idf(term: str, document: str, corpus: List):
    """ Term Frequency-Inverse Document Frequency.
    """
    return tf(term, document) * idf(term, corpus)


def train_test_split(dataset: List, ratio: float)-> Tuple[List]:
    """ Simple train test split of a dataset,
    that is a namedtuple.
    """
    shuffle(dataset)
    split_ratio = ratio
    split_index = int(split_ratio * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data


def clean_dataset(data: Dict[str, str], stopwords: List = [])-> Dict[str, List[str]]:
    """ Takes a string value in a dict into a list of words, lowers them and only keeps
    those that are only letters.
    """
    cleaned = []
    for dict in data:
        text = []
        for key, value in dict.items():
            for word in value.split(" "):
                if word.isalpha() and word not in stopwords:
                    text.append(word.lower())
        cleaned.append({key: text})
    return cleaned
