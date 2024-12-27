import sys
sys.path.append("../")
import numpy as np
from numpy.linalg import norm
from typing import List, Tuple
from collections import namedtuple
from utils import build_dataset, tf_idf, train_test_split, idf, tf

from sklearn.metrics.pairwise import cosine_similarity


class KNN():

    def __init__(self,points, k):
        self.k = k
        self.points = points


    def cosine_similarity(self, a, b):
        """

        """
        dot_product = np.dot(a, b)
        return dot_product/ (norm(a)*norm(b))


    def classify_point(self,  point):
        """

        """ 
        distance = []

        for tup in self.points:
            dist = cosine_similarity(point, tup[0])
            distance.append((dist, tup[1].category))

        distance = sorted(distance)[:self.k]
        freq1 = 0 
        freq2 = 0 

        for d in distance:
            if d[1] == 'ham':
                freq1 += 1 
            elif d[1] == 'spam':
                freq2 += 1 
        return 'ham' if freq1 > freq2 else "spam"





def build_tf_idf_matrix(corpus: List[namedtuple])-> Tuple[List, List]:
    """

    """

    vocab = sorted({word.lower() for doc in corpus for word in doc.text.split() if word.isalpha()})
    
    idf_dict = {term: idf(term, corpus) for term in vocab} 
    tf_idf_matrix = []
    for document in corpus:
        vector = [0] * len(vocab)

        for i, term in enumerate(vocab):
            vector[i] = tf(term, document.text) * idf_dict[term]

        tf_idf_matrix.append(vector)

    return tf_idf_matrix, vocab


Email = namedtuple('Email', "category, text")
path_to_data = "../data/emails/email.csv"

dataset = build_dataset(path_to_data, Email)



tf_idf_matrix, vocab = build_tf_idf_matrix(dataset[:500])
encoded_data = list(zip(tf_idf_matrix, dataset[:500]))
train_data, test_data = train_test_split(encoded_data, 0.8)

clf = KNN(train_data, k=5)
correct = 0

for point in test_data:
    prediction = clf.classify_point(point[0])
    if prediction == point[1].category:
        correct += 1 

print(f"Accuracy is : {correct / len(test_data)}")
