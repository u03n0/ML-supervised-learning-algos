import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from numpy.linalg import norm
import numpy as np
from utils.py_utils  import build_dataset, train_test_split, clean_dataset, idf, tf
from numba import jit
from config import BASE_DIR, DATA_PATH


def build_tf_idf_matrix(corpus: list[dict[str, list[str]]]) -> np.ndarray:
    """
    Builds a TF-IDF matrix from a corpus of documents.
    Each row represents a document, and each column represents a term.
    """
    # Step 1: Create a sorted vocabulary from the corpus
    vocab: set = sorted(set(word for doc in corpus for word in doc.values()))
    vocab_size: int = len(vocab)
    num_documents: int = len(corpus)
    idf_dict: dict = {term: idf(term, corpus) for term in vocab}
    # Step 2: Initialize a zero matrix with dtype=np.float32
    tf_idf_matrix: np.ndarray = np.zeros((num_documents, vocab_size), dtype=np.float32)

    # Step 3: Populate the matrix with TF-IDF values
    for doc_index, document in enumerate(corpus):
        for term_index, term in enumerate(vocab):
            tf_idf_matrix[doc_index, term_index] = tf(term, document.values()) * idf_dict[term]

    return tf_idf_matrix

@jit(nopython=True)
def cosine_similarity(a, b) -> float:
    """ The dot product of two vectors (a, b) divided
    by the product of the magnitudes of the vectors.
    cosine similarity = A * B / ||A|| * ||B||
    """
    dot_product = np.dot(a, b)
    return dot_product / (norm(a)* norm(b))


def classify_point(dataset: list[dict[str, list[str]]], point, k: int = 3)-> str:
    """ Finds the top K cosine similarities from the point to all other
    points in the dataset. A point is a vector representation of a document.
    """
    distance = []
    for tup in dataset:
        array, dict = tup
        label = list(dict.keys())[0]
        dist = cosine_similarity(point, array)
        distance.append((dist, label))

    distance = sorted(distance)[:k]
    freq1 = 0 
    freq2 = 0 

    for d in distance:
        _, label = d
        if label == 'ham':
            freq1 += 1 
        elif label == 'spam':
            freq2 += 1 
    return 'ham' if freq1 > freq2 else "spam"



# Load data from csv file
path_to_data: Path = BASE_DIR / DATA_PATH / "emails" / "email.csv"
with open(Path(BASE_DIR / DATA_PATH / "stopwords_en.txt"), 'r') as file:
    stopwords: list = [file.read().replace('\n', ',')]
 
dataset = build_dataset(path_to_data)
# Clean dataset
clean = clean_dataset(dataset, stopwords)
# Build tf_idf matrix map
tf_idf_matrix = build_tf_idf_matrix(dataset)
# Newly encoded dataset
encoded_data: list = list(zip(tf_idf_matrix, clean))
# Train test split
train_data, test_data = train_test_split(encoded_data, 0.8)
# Make predictions
correct: int = 0
for example in test_data:
    vector, dict = example
    y_pred = list(dict.keys())[0]
    y_hat = classify_point(train_data, vector, k=5)
    if y_hat == y_pred:
        correct += 1 

print(f"accuracy is : {correct / len(test_data)}")


    
