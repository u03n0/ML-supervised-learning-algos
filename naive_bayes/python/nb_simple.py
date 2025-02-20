import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.py_utils import build_dataset, train_test_split, clean_dataset
from config import BASE_DIR, DATA_PATH


start_time = time.time()


def build_historgram(dataset: list[dict[str, list[str]]])-> dict[str, int]:
    """ Builds a histogram, which is a dict with keys being unique words
    and values being the occurence of said word in that category (ham, spam)
    """
    word_dict: dict = {}
    for dict_ in dataset:
        for word_list in dict_.values():
            for word in word_list:
                if word not in word_dict:
                    word_dict[word] = 1 
                else:
                    word_dict[word] += 1 
    return word_dict

def compute_product(text: dict[str, list[str]], histogram: dict[str, int], num_words: int, proba: float, alpha: int = 1)-> float:
        """ The product of a document is computed by multiplying the histogram value for each term. Finally the 
        pronbability of a given class is also multiplied. 

        An alpha value is used incase a word is not present in a histogram, to be consistant, it is added to the value 
        if a word is found in a histogram.
            Ex: document 'The dog' = 'The' (0.8922)+alpha * 'dog' (0.034)+alpha * probability of ham (0.84) = 1.643489
        """
        product: float = 1.0 
        for word_list in text.values():
            for word in word_list:
                if word in histogram:
                    product *= ((histogram[word] + alpha ) / num_words)
                else:
                    product *= alpha / num_words
        return product * proba


path: Path = BASE_DIR / DATA_PATH / "emails" / "email.csv"
dataset: list[dict[str, str]] = build_dataset(path)
clean: list[dict[str, list[str]]] = clean_dataset(dataset)
train_data, test_data = train_test_split(clean, 0.8)  # Train test split_index

len_ham: int = sum([1 for dict in train_data if 'ham' in dict])

ham_proba: float = len_ham / len(train_data) 
spam_proba: float = 1.0 - ham_proba

# Make histograms 
spam_histo: dict[str, int] = build_historgram([dict for dict in train_data if 'spam' in dict])
ham_histo: dict[str, int] = build_historgram([dict for dict in train_data if 'ham' in dict])

# Total words in spam and ham
num_spam_words: int = sum(spam_histo.values())
num_ham_words: int = sum(ham_histo.values())
# Evaluation
results: list = []
for email in test_data:
    y_pred: str = list(email.keys())[0]
    product_ham: float = compute_product(email, ham_histo, num_ham_words, ham_proba)
    product_spam: float = compute_product(email, spam_histo, num_spam_words, spam_proba)
    results.append(("spam" if product_spam > product_ham else "ham", y_pred))
    
correct: int = 0 
for y_hat, y_pred in results:
    if y_hat == y_pred:
        correct += 1 

print(f"the accuracy is {correct / len(test_data)}")
end_time: float = time.time()
execution_time: float = end_time - start_time
print(f"Execution time: {execution_time} milliseconds")
