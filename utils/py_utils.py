from random import shuffle
from typing import List, Dict, Tuple
from math import log
from pathlib import Path


def build_dataset(path: str)-> list[dict[str, str]]:
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

    
def tf(term: str, document: str):
    """ Term Frequency calculator.
    Relative frequency of term t within a document d 
    """
    n: int = len(document)
    freq: int = 0
    for word in document:
        if word == term:
            freq += 1
    if freq:
        return (freq / n) + 1
    else:
        return 1 / n 


def idf(term: str, corpus: list[dict[str, list[str]]])-> float:
    """ Inverse Document Frequency. Log of number of 
    documents (N) in corpus (D) over 1 + the number of 
    documents (d) within the corpus (D) that the term (t)
    appears in. 
    """
    n: int = len(corpus)
    docs_with_term_count: int = sum(1 for doc in corpus if term in doc.values())
    return log(n / (1 + docs_with_term_count)) + 1


def tf_idf(term: str, document: str, corpus: list[dict[str, list[str]]])-> float:
    """ Term Frequency-Inverse Document Frequency.
    """
    return tf(term, document) * idf(term, corpus)


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


def clean_dataset(data: list[dict[str, str]], stopwords: list = [])-> list[dict[str, list[str]]]:
    """ Takes a string value in a dict into a list of words, lowers them and only keeps
    those that are only letters.
    """
    cleaned: list = []
    for dict in data:
        text: list = []
        for key, value in dict.items():
            for word in value.split(" "):
                if word.isalpha() and word not in stopwords:
                    text.append(word.lower())
        cleaned.append({key: text})
    return cleaned
