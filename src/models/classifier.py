import pandas as pd
from transformers import  AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.features import build_dataset, load_model_tokenizer, get_features
from transformers import BertTokenizer, BertForSequenceClassification

import torch



def run_model(df: pd.DataFrame):
    
    output_dir = "../../models/fine_tuned_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_tokenizer(output_dir, device)
    model.to(device)

    dataset = build_dataset(tokenizer, get_features(df), device)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Fine-tune the model
    optimizer = AdamW(model.parameters(), lr=1e-5)
    epochs = 3

    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Evaluate the model
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1)

            predictions.extend(predicted_labels.gpu().numpy())
            true_labels.extend(batch[2].gpu().numpy())

    
    model.save_pretrained(output_dir)

    return predictions, true_labels
