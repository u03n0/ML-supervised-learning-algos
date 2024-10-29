import sys
sys.path.append("../")

from collections import namedtuple

from naive_bayes.utils import build_dataset


Email = namedtuple('Email', "category, text")
path_to_data = "../data/emails/email.csv"

dataset = build_dataset(path_to_data, Email)

print(dataset[0])






