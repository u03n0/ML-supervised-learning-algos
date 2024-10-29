import csv
from typing import Tuple, List
from collections import namedtuple
from random import shuffle


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


