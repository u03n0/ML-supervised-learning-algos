import sys
sys.path.append("../")
from typing import List, Tuple
from numpy.linalg import norm
import numpy as np
from collections import namedtuple
from utils import build_dataset, tf_idf, train_test_split


Email = namedtuple('Email', "category, text")
path_to_data = "../data/emails/email.csv"

dataset = build_dataset(path_to_data, Email)

def build_tf_idf_matrix(corpus: List[namedtuple])-> Tuple[List, List]:
    """

    """

    vocab = sorted(set(word for doc in corpus for word in doc.text.split()))

    tf_idf_matrix = []
    for document in corpus:
        vector = [0] * len(vocab)

        for i, term in enumerate(vocab):
            vector[i] = tf_idf(term, document.text, corpus)

        tf_idf_matrix.append(vector)

    return tf_idf_matrix, vocab

def cosine_similarity(a, b):
    """

    """
    dot_prouct = np.dot(a, b)
    return dot_prouct/ (norm(a)*norm(b))



def classify_point(dataset, point, k=3):
    
    distance = []
    for tup in dataset:
        dist = cosine_similarity(point, tup[0])
        distance.append((dist, tup[1].category))

    distance = sorted(distance)[:k]
    freq1 = 0 
    freq2 = 0 

    for d in distance:
        if d[1] == 'ham':
            freq1 += 1 
        elif d[1] == 'spam':
            freq2 += 1 
    return 'ham' if freq1 > freq2 else "spam"




            
tf_idf_matrix, vocab = build_tf_idf_matrix(dataset[:300])
small_data = dataset[:300]
encoded_data = list(zip(tf_idf_matrix, small_data))
train_data, test_data = train_test_split(encoded_data, 0.8)

correct = 0
for example in test_data:
    if classify_point(train_data, example[0], k=5) == example[1].category:
        correct += 1 

print(f"accuracy is : {correct / len(test_data)}")


    
