import pandas as pd
from typing import Optional
from pathlib import Path


def save_data(df: pd.DataFrame, sub_folder: str)-> Optional[None]:
    """ Saves a Pandas DataFrame to '/data/sub_folder/'
    where subfolder can be 'processed', 'interim', or 'raw'
    """
    full_path = create_abs_path(sub_folder)
    df.to_csv(full_path, index=False)
    return 
    
def build_filename(prefix: str)-> str:
    """ creates the appropriate file name for a df
    """
    path = '../data/raw/internship_challenge - dataset.csv'
    filename = path.split('/')[-1]

    return prefix + "_" + filename

def create_abs_path(sub_folder):
    """Builds an absolute path for a csv file
    within /data/particular_sub_folder/
    """
    relative_path = f"../data/{sub_folder}/"
    abs_path = Path(relative_path).resolve()
    abs_path.mkdir(parents=True, exist_ok=True)
    return abs_path / build_filename(sub_folder)
    
