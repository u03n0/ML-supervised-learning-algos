from typing import List, Dict

from utils import build_dataset, train_test_split, clean_dataset


def build_historgram(dataset: List[Dict[str, List[str]]])-> Dict[str, int]:
    """ Builds a histogram, which is a dict with keys being unique words
    and values being the occurence of said word in that category (ham, spam)
    """
    word_dict = {}
    for dict in dataset:
        for word_list in dict.values():
            for word in word_list:
                if word not in word_dict:
                    word_dict[word] = 1 
                else:
                    word_dict[word] += 1 
    return word_dict

def compute_product(text: Dict[str, List[str]], histogram: Dict[str, int], num_words: int, proba: float, alpha: int = 1):
        """ The product of a document is computed by multiplying the histogram value for each term. Finally the 
        pronbability of a given class is also multiplied. 

        An alpha value is used incase a word is not present in a histogram, to be consistant, it is added to the value 
        if a word is found in a histogram.
            Ex: document 'The dog' = 'The' (0.8922)+alpha * 'dog' (0.034)+alpha * probability of ham (0.84) = 1.643489
        """
        product = 1 
        for word_list in text.values():
            for word in word_list:
                if word in histogram:
                    product *= ((histogram[word] + alpha ) / num_words)
                else:
                    product *= alpha / num_words
        return product * proba


# Create the dataset from a csv file
email_path = '../data/emails/email.csv'
dataset = build_dataset(email_path)
# Clean the data
clean = clean_dataset(dataset)
# Train test split_index
train_data, test_data = train_test_split(clean, 0.8) 
# Get probabilities of ham and spam given total emails
len_ham = sum([1 for dict in train_data if 'ham' in dict])
ham_proba = len_ham / len(train_data) 
spam_proba = 1.0 - ham_proba

# Make histograms 
spam_histo = build_historgram([dict for dict in train_data if 'spam' in dict])
ham_histo = build_historgram([dict for dict in train_data if 'ham' in dict])

# Total words in spam and ham
num_spam_words = sum(spam_histo.values())
num_ham_words = sum(ham_histo.values())
# Evaluation
results = []
for email in test_data:
    y_pred = list(email.keys())[0]
    product_ham = compute_product(email, ham_histo, num_ham_words, ham_proba)
    product_spam = compute_product(email, spam_histo, num_spam_words, spam_proba)
    results.append(("spam" if product_spam > product_ham else "ham", y_pred))
    
correct = 0 
for y_hat, y_pred in results:
    if y_hat == y_pred:
        correct += 1 

print(f"the accuracy is {correct / len(test_data)}")
