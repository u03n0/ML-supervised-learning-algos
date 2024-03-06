import pandas as pd
from typing import Callable


def process_interim(df: pd.DataFrame)-> Callable:
    new = df.copy()
    return apply_lowering(new)


def apply_lowering(df: pd.DataFrame)-> Callable:
    new = df.copy()
    new['text'] = new['text'].str.lower()
    return apply_remove_extra_quotations(new)


def remove_extra_quotations(text:str)-> str:
    """ removes extra quotation marks and \n found within text.
    """
    alist = text.split('"')
    clean = [word.replace("\n", '') for word in alist]
    return ''.join(clean)

def apply_remove_extra_quotations(df: pd.DataFrame)-> pd.DataFrame:
    new = df.copy()
    new['text'] = new['text'].apply(remove_extra_quotations)
    return new