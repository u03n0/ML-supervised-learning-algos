import time
from collections import namedtuple

from utils import build_dataset, train_test_split, clean_dataset


start = time.time()
Email = namedtuple('Email', 'category, text')
email_path = '../data/emails/email.csv'



class NaiveBayesClassifier():

    def __init__(self):
        pass

    def fit(self, train_data):
        self.hams = [email for email in train_data if email.category == 'ham']
        self.spams = [email for email in train_data if email.category == 'spam']
        self.ham_proba = len(self.hams) / len(train_data)
        self.spam_proba = 1.0 - self.ham_proba
        self.ham_histo = self.get_historgram("ham")
        self.spam_histo = self.get_historgram("spam")
        self.num_ham_words = sum(self.ham_histo.values())
        self.num_spam_words = sum(self.spam_histo.values())
       
    def predict(self, test_data):
        results = []
        for email in test_data:
            product_ham = self.compute_product(email.text, 'ham')
            product_spam = self.compute_product(email.text, 'spam')
            results.append("spam" if product_spam > product_ham else "ham")
        return results

    def compute_product(self, text, category):
        product = 1 
        histogram = self.ham_histo if category == 'ham' else self.spam_histo
        num_words = self.num_ham_words if category == 'ham' else self.num_spam_words
        proba = self.ham_proba if category == 'ham' else self.spam_proba

        for word in text:
            if word in histogram:
                product *= ((histogram[word] + 1) / num_words)
            else:
                product *= 1 / num_words
        return product * proba


    def get_historgram(self, category):
        """ Builds a histogram, which is a dict with keys being unique words
        and values being the occurence of said word in that category (ham, spam)
        """
        word_dict = {}
        dataset = self.hams if category == 'ham' else self.spams
        for email in dataset:
            for word in email.text:
                if word not in word_dict:
                    word_dict[word] = 1 
                else:
                    word_dict[word] += 1 

        return word_dict


# Create the dataset from a csv file
dataset = build_dataset(email_path, Email)
# Clean the data
cleaned = clean_dataset(dataset, Email)
# Train test split_index
train_data, test_data = train_test_split(cleaned, 0.8) 
model = NaiveBayesClassifier()
model.fit(train_data)
predictions = model.predict(test_data)
# Get counts
correct = 0 
for prediction in zip(predictions, test_data):
    if prediction[0] == prediction[1].category:
        correct += 1 

print(f"the accuracy is {correct / len(test_data)}")
print(time.time()-start)
