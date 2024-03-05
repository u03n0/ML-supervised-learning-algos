import re
import emoji

from typing import List, Dict, Callable
import pandas as pd


def clean_data(df: pd.DataFrame):
    pass


def reduce_dataframe(df: pd.DataFrame)-> Callable:
    new = df[['id', 'about', 'keywords', 'specialities_x', 'Label']]
    return fix_na_specialities_x(new)

def fix_na_specialities_x(df: pd.DataFrame)-> pd.DataFrame:
    new = df.copy()
    new['specialities_x'] =  new['specialities_x'].fillna('')
    return new

def remove_emojis(text: str)-> str:
    """ removes emojis and :emoji: syntax from text.
    """
    no_emojis_text = emoji.demojize(text)
    cleaned_text = re.sub(r':[^:]+:', '', no_emojis_text)
    return cleaned_text

def apply_remove_emojis(df: pd.DataFrame)-> pd.DataFrame:
    new = df.copy()
    new['about'] = new['about'].apply(remove_emojis)
    return new

def remove_extra_quotations(text:str)-> str:
    """ removes extra quotation marks found within text.
    """
    alist = text.split('"')
    return ''.join(alist)

def apply_remove_extra_quotations(df: pd.DataFrame)-> pd.DataFrame:
    new = df.copy()
    new['about'] = new['about'].apply(remove_extra_quotations)
    return new

def convert_to_list(text:str)-> List[str]:
    """ Removal of unwanted words and characters from strings.
    Finally, conversion to a list.
    """
    return text.replace('and', '').replace("[", '').replace("]", '').replace("'", '').split(',')
