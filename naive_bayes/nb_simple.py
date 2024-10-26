from random import sample
import csv
from collections import namedtuple
from typing import List, Dict


Email = namedtuple('Email', 'category, text')
email_path = '../data/emails/email.csv'


def build_dataset(path: str, nt: namedtuple)-> namedtuple:
    """ Reads csv file and builds dataset as a list of namedtuples
    """
    with open(path, 'r') as file:
        return [nt(*row) for row in csv.reader(file) if all(row)][1:]


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

# Create the dataset from a csv file
dataset = build_dataset(email_path, Email)
# Clean the data
clean_data = clean_dataset(dataset, Email)
# Train test split_index
shuffled_data = sample(clean_data, len(clean_data))
split_ratio = 0.8
split_index = int(split_ratio * len(shuffled_data))
train_data = shuffled_data[:split_index]
test_data = shuffled_data[split_index:]
# Get counts
ham_count = len([email for email in train_data if email.category == 'ham'])
total_emails = len(train_data)
spam_count = total_emails - ham_count
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
    product_ham = sum([ham_histo[word] / num_ham_words for word in email.text if word in ham_histo]) * ham_proba
    product_spam = sum([spam_histo[word]/ num_spam_words for word in email.text if word in spam_histo]) * spam_proba

    if product_ham > product_spam:
        results.append('ham')
    else:
        results.append('spam')

correct = 0 
for prediction in zip(results, test_data):
    if prediction[0] == prediction[1].category:
        correct += 1 

print(f"the accuracy is {correct / len(test_data)}")
