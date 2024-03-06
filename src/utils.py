import pandas as pd
from typing import Optional, Boolean
from pathlib import Path



def save_data(df: pd.DataFrame, sub_folder: str)-> Optional[None]:
    """ Saves a Pandas DataFrame to '/data/sub_folder/'
    where subfolder can be 'processed', 'interim', or 'raw'
    """
    full_path = create_abs_path(df, sub_folder)
    df.to_csv(full_path, index=False)
    return 
    
def build_filename(df: pd.DataFrame, prefix: str)-> str:
    """ creates the appropriate file name for a df
    """
    return prefix + "_" + df.file_name

def get_fullpath(df: pd.DataFrame, sub_folder: str):
    """ Determines if a csv file exists within /data/particular_sub_folder/
    """
    full_path = create_abs_path(df, sub_folder)
    return full_path

def create_abs_path(df, sub_folder):
    """Builds an absolute path for a csv file
    within /data/particular_sub_folder/
    """
    relative_path = f"data/{sub_folder}/"
    abs_path = Path(relative_path).resolve()
    abs_path.mkdir(parents=True, exist_ok=True)
    return abs_path / build_filename(df, sub_folder)
    
