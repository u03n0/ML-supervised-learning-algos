



class KNN():

    def __init__(self,points, k):
        self.k = k
        self.points = points


    def cosine_similarity(self, a, b):
        """

        """
        dot_prouct = np.dot(a, b)
        return dot_prouct/ (norm(a)*norm(b))

    def fit(self):
        pass

    def predict(self):
        pass




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





