import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset
from typing import Tuple, List
import torch


def load_model_tokenizer(output_dir, device):
    """ Load tokenizer and pre-trained BERT model or create new ones if not saved
    """
    try:
        model = BertForSequenceClassification.from_pretrained(output_dir).to(device)
        tokenizer = BertTokenizer.from_pretrained(output_dir)
        print("Model and tokenizer loaded from", output_dir)
    except:
        print("Loading failed. Creating new model and tokenizer.")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return tokenizer, model

def build_dataset(tokenizer, a_tuple, device):
    texts, labels =  a_tuple
    tokenized_data = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    input_ids = tokenized_data['input_ids'].to(device)
    attention_mask = tokenized_data['attention_mask'].to(device)
    labels = torch.tensor(labels).to(device)

    return TensorDataset(input_ids, attention_mask, labels)

def get_features(df: pd.DataFrame)-> Tuple[List[str], List[int]]:
    """ returns a list of 'text' and 'labels'
    """
    return df['text'].tolist(), df['label'].tolist()