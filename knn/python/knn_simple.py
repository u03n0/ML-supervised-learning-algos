import sys
import time
import random
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.py_utils  import build_dataset, train_test_split, clean_dataset, get_tf_idf, metric_
from config import BASE_DIR, DATA_PATH


start_time = time.time()

parser = argparse.ArgumentParser() # Argument Parser
parser.add_argument(
        '--metric', 
        choices=['euclidean', 'cosine'], 
        required=True, 
        help="Choose the distance metric: 'euclidean' or 'cosine'"
    )
args = parser.parse_args()


def classify_point(dataset: list[dict[str, list[float]]], point: list[float], k: int = 3)-> str:
    """ Finds the top K cosine similarities from the point to all other
    points in the dataset. A point is a vector representation of a document.
    """
    distance: list = []
    for dict_ in dataset:
        for label, vector in dict_.items():
            dist = metric_(point, vector, args.metric)
            distance.append((dist, label))
    distance = sorted(distance)[:k] if args.metric == 'cosine' else sorted(distance, reverse=True)[:k]
    freq1: int = 0 
    freq2: int = 0 

    for item in distance:
        _, label = item
        if label == 'ham':
            freq1 += 1 
        elif label == 'spam':
            freq2 += 1 
    if freq1 == freq2:
        return random.choice(["ham", "spam"])
    return 'ham' if freq1 > freq2 else "spam"



with open(Path(BASE_DIR / DATA_PATH / "stopwords_en.txt"), 'r') as file:
    stopwords: list = [file.read().replace('\n', ',')]
 

path_to_data: Path = BASE_DIR / DATA_PATH / "emails" / "email.csv"
dataset = build_dataset(path_to_data)
clean = clean_dataset(dataset[:200])
encoded_data = get_tf_idf(clean)
# Train test split
train_data, test_data = train_test_split(encoded_data, 0.8)
# Make predictions
correct: int = 0
for dict_ in test_data:
    for label, vector in dict_.items():
        y_pred = label
        y_hat = classify_point(train_data, vector, k=9)
        if y_hat == y_pred:
            correct += 1 

print(f"accuracy is : {correct / len(test_data)} using {args.metric}.")

end_time: float = time.time()
execution_time: float = end_time - start_time
print(f"Execution time: {execution_time} milliseconds")

