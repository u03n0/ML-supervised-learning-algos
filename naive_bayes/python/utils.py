from typing import Tuple, List, Dict
from random import shuffle


def train_test_split(dataset: List[Dict[str, List[str]]], ratio: float)-> Tuple[List[Dict]]:
    """ Simple train test split of a dataset based on a ration of how many 
    to be used for trianing, and remainder for testing.
    """
    shuffle(dataset)
    split_index = int(ratio * len(dataset))
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    return train_data, test_data


def clean_dataset(data: Dict[str, str])-> Dict[str, List[str]]:
    """ Takes a string value in a dict into a list of words, lowers them and only keeps
    those that are only letters.
    """
    cleaned = []
    for dict in data:
        text = []
        for key, value in dict.items():
            for word in value.split(" "):
                if word.isalpha():
                    text.append(word.lower())
        cleaned.append({key: text})
    return cleaned


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
