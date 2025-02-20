import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.py_utils import build_dataset,train_test_split, clean_dataset, get_tf_idf, cosine_similarity
from config import BASE_DIR, DATA_PATH


start_time = time.time()

class KNN():

    def __init__(self, points, k):
        self.k = k
        self.points = points

    def classify_point(self,  point):
        """

        """ 
        distance = []

        for dict_ in self.points:
            for label, vector in dict_.items():
                dist = cosine_similarity(point, vector)
                distance.append((dist, label))

        distance = sorted(distance)[:self.k]
        freq1 = 0 
        freq2 = 0 

        for d in distance:
            if d[1] == 'ham':
                freq1 += 1 
            elif d[1] == 'spam':
                freq2 += 1 
        return 'ham' if freq1 > freq2 else "spam"



path_to_data = BASE_DIR / DATA_PATH / "emails" / "email.csv"
dataset = build_dataset(path_to_data)
cleaned_dataset = clean_dataset(dataset[:300])
encoded_data = get_tf_idf(cleaned_dataset)

train_data, test_data = train_test_split(encoded_data, 0.8)

clf = KNN(train_data, k=5)
correct = 0

for dict_ in test_data:

    for label, vector in dict_.items():
        prediction = clf.classify_point(vector)
        if prediction == label:
            correct += 1 

print(f"Accuracy is : {correct / len(test_data)}")

end_time: float = time.time()
execution_time: float = end_time - start_time
print(f"Execution time: {execution_time} milliseconds")