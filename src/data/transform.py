import pandas as pd
from utils import create_abs_path, save_data
from data.cleaning import clean_data
from data.preprocessing import process_interim


def transform_data(df: pd.DataFrame)-> pd.DataFrame:
    """ Applies cleaning to raw data, or
    preprocessing to interim data, or
    returns pre-processed data
    """

    processed_path = create_abs_path('processed')

    if processed_path.exists():
        return pd.read_csv(processed_path)
    
    elif create_abs_path('interim').exists():
        interim_path = create_abs_path('interim')
        current_df = pd.read_csv(interim_path)
        processed_df = process_interim(current_df)
        save_data(processed_df, 'processed')
        return transform_data(df)
    
    else:
        cleaned_df = clean_data(df)
        save_data(cleaned_df, 'interim')
        return transform_data(df)