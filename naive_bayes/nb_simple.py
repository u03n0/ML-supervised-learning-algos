import time
from random import shuffle
from collections import namedtuple
from typing import List, Dict

from utils import build_dataset, train_test_split


start = time.time()
Email = namedtuple('Email', 'category, text')
email_path = '../data/emails/email.csv'


def clean_dataset(data: List[namedtuple], nt: namedtuple)-> namedtuple:
    """ Breaks Email.text into a list of words, lowers them and only keeps
    those that are only letters.
    """
    cleaned = []
    for email in data:
        text = []
        for word in email.text.split():
            if word.isalpha():
                text.append(word.lower())
        cleaned.append(Email(email.category, text))
    return cleaned


def get_historgram(dataset: List[namedtuple], category: str)-> Dict[str, int]:
    """ Builds a histogram, which is a dict with keys being unique words
    and values being the occurence of said word in that category (ham, spam)
    """
    word_dict = {}
    for named_tup in dataset:
        if named_tup.category == category:
            for word in named_tup.text:
                if word not in word_dict:
                    word_dict[word] = 1 
                else:
                    word_dict[word] += 1 

    return word_dict

def compute_product(text, histogram, num_words, proba):
        product = 1 
        for word in text:
            if word in histogram:
                product *= ((histogram[word] +1 ) / num_words)
            else:
                product *= 1 / num_words
        return product * proba


# Create the dataset from a csv file
dataset = build_dataset(email_path, Email)
# Clean the data
clean_data = clean_dataset(dataset, Email)
# Train test split_index
train_data, test_data = train_test_split(clean_data, 0.8) 
# Get counts
ham_count = len([email for email in train_data if email.category == 'ham'])
total_emails = len(train_data)
# Get probabilities of ham and spam given total emails
ham_proba = ham_count / total_emails 
spam_proba = 1.0 - ham_proba
# Make histograms / 
spam_histo = get_historgram(train_data, 'spam')
ham_histo = get_historgram(train_data, 'ham')
# Total words in spam and ham
num_spam_words = sum(spam_histo.values())
num_ham_words = sum(ham_histo.values())
# Evaluation
results = []
for email in test_data:
    product_ham = compute_product(email.text, ham_histo, num_ham_words, ham_proba)
    product_spam = compute_product(email.text, spam_histo, num_spam_words, spam_proba)
    results.append("spam" if product_spam > product_ham else "ham")
    
correct = 0 
for prediction in zip(results, test_data):
    if prediction[0] == prediction[1].category:
        correct += 1 

print(f"the accuracy is {correct / len(test_data)}")
print(time.time())
