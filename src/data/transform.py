import pandas as pd
from src.utils import get_fullpath, save_data
from cleaning import clean_data
from preprocessing import process_interim


def transform_data(df: pd.DataFrame)-> pd.DataFrame:
    """ Applies cleaning to raw data, or
    preprocessing to interim data, or
    returns pre-processed data
    """

    processed = get_fullpath(df, 'processed')

    if processed.exists():
        return pd.read_csv(processed)
    
    elif get_fullpath(df, 'interim').exists():
        interim = get_fullpath(df, 'interim')
        current_df = pd.read_csv(interim)
        processed = process_interim(current_df)
        save_data(processed, 'processed')
        return transform_data(df)
    
    else:
        cleaned_df = clean_data(df)
        save_data(cleaned_df, 'interim')
        return transform_data(df)