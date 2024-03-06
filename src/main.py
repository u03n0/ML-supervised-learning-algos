import pandas as pd
from data.transform import transform_data
from models.classifier import run_model


path = '../data/raw/internship_challenge - dataset.csv'

df = pd.read_csv(path)


fs_df = transform_data(df)


predictions, true_labels = run_model(fs_df)



