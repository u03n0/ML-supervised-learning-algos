import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.py_utils import build_dataset, train_test_split, clean_dataset
from config import BASE_DIR, DATA_PATH


start_time = time.time()


class NaiveBayesClassifier():
    """ Naive Bayes Classifier.
    """

    def __init__(self, alpha: int =1):
        self.alpha = alpha


    def fit(self, train_data: list[dict[str, list[str]]]):
        """ Calculates histograms and probabilities need later for prediction.
        """
        self.hams: list[dict[str, list[str]]] = [dict for dict in train_data if 'ham' in dict]
        self.spams: list[dict[str, list[str]]] = [dict for dict in train_data if 'spam' in dict]
        self.ham_proba: float = len(self.hams) / len(train_data)
        self.spam_proba: float = 1.0 - self.ham_proba
        self.ham_histo = self.get_historgram("ham")
        self.spam_histo = self.get_historgram("spam")
        self.num_ham_words = sum(self.ham_histo.values())
        self.num_spam_words = sum(self.spam_histo.values())
       
    def predict(self, test_data: list[dict[str, list[str]]]):
        """ Takes test data and predicts the class for a given 
        email (text).
        """
        results: list = []
        for email in test_data:
            y_pred = list(email.keys())[0]
            product_ham = self.compute_product(email, 'ham')
            product_spam = self.compute_product(email, 'spam')
            results.append(("spam" if product_spam > product_ham else "ham", y_pred))
        return results

    def compute_product(self, dict, category):
        """ Helper method to calculate the product of a text (email),
        this is done by multiplying the histogram of each word , diviided by the number of words and 
        finally multiplying it by a probability.
        """
        product = 1 
        histogram = self.ham_histo if category == 'ham' else self.spam_histo
        num_words = self.num_ham_words if category == 'ham' else self.num_spam_words
        proba = self.ham_proba if category == 'ham' else self.spam_proba

        for word_list in dict.values():
            for word in word_list:
                if word in histogram:
                    product *= ((histogram[word] + self.alpha) / num_words)
                else:
                    product *= self.alpha / num_words
        return product * proba


    def get_historgram(self, category):
        """ Builds a histogram, which is a dict with keys being unique words
        and values being the occurence of said word in that category (ham, spam)
        """
        word_dict = {}
        dataset = self.hams if category == 'ham' else self.spams
        for document in dataset:
            for word_list in document.values():
                for word in word_list:
                    if word not in word_dict:
                        word_dict[word] = 1 
                    else:
                        word_dict[word] += 1 
        return word_dict


# Create the dataset from a csv file
path: Path = BASE_DIR / DATA_PATH / "emails" / "email.csv"

dataset = build_dataset(path)
# Clean the data
cleaned = clean_dataset(dataset)
# Train test split_index
train_data, test_data = train_test_split(cleaned, 0.8) 
# Intialize classifier
model = NaiveBayesClassifier()
model.fit(train_data)
predictions = model.predict(test_data)
# Get counts
correct = 0 
for y_hat, y_pred in predictions:
    if y_hat == y_pred:
        correct += 1 

print(f"the accuracy is {correct / len(test_data)}")
end_time: float = time.time()
execution_time: float = end_time - start_time
print(f"Execution time: {execution_time} milliseconds")