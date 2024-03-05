import pandas as pd
from typing import Optional
from pathlib import Path



def save_data(df: pd.DataFrame, filename:str='clean.csv')-> Optional[None]:
    """ Saves a Pandas DataFrame to '/data/processed/'.
    Filename can be any name but must have a .csv extension.
    """

    relative_path = "data/interim/"
    abs_path = Path(relative_path).resolve()
    abs_path.mkdir(parents=True, exist_ok=True)
    full_path = abs_path / filename
    _, ext = filename.split(".")
    if ext.lower() == 'csv':
        df.to_csv(full_path, index=False)
        return 
    else:
        raise ValueError("File must have a .csv extension.")