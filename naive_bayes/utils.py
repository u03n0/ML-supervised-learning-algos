import csv
from typing import Tuple, List
from collections import namedtuple
from random import shuffle


def build_dataset(path: str, nt: namedtuple)-> namedtuple:
    """ Reads csv file and builds dataset as a list of namedtuples
    """
    with open(path, 'r') as file:
        return [nt(*row) for row in csv.reader(file) if all(row)][1:]

def train_test_split(dataset: List[namedtuple], ratio: float)-> Tuple[List[namedtuple]]:
    """ Simple train test split of a dataset,
    that is a namedtuple.
    """
    shuffle(dataset)
    split_ratio = 0.8
    split_index = int(split_ratio * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data

def mul(l: List[float|int]):
    """ Multiplies all elements in a list.
    """
    product = 1
    for item in l:
        product *= item
    return product


def clean_dataset(data: List[namedtuple], Nt: namedtuple)-> namedtuple:
    """ Breaks Email.text into a list of words, lowers them and only keeps
    those that are only letters.
    """
    cleaned = []
    for email in data:
        text = []
        for word in email.text.split():
            if word.isalpha():
                text.append(word.lower())
        cleaned.append(Nt(email.category, text))
    return cleaned


