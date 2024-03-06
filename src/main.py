import pandas as pd
from data.transform import transform_data
from models.classifier import run_model
from visualization import plot_confusion_matrix


path = '../data/raw/internship_challenge - dataset.csv'

df = pd.read_csv(path)
# data is read to be used in model
fs_df = transform_data(df)
# model is loaded, fine-tuned, evaluated and saved
predictions, true_labels = run_model(fs_df)

print(predictions, true_labels)
# Visualization of results
plot_confusion_matrix(predictions, true_labels)