import re
import emoji

from typing import List, Callable

import pandas as pd




def clean_data(df: pd.DataFrame)-> Callable:
    """ Begins process of applying steps
    in order to clean the raw data into an
    interim state.
    """
    new = df.copy()
    return reduce_dataframe(new)

def reduce_dataframe(df: pd.DataFrame)-> Callable:
    """ builds new df from pre-selected columns
    """
    new = df[['id', 'about', 'keywords', 'specialities_x', 'Label']]
    return fix_na_specialities_x(new)

def fix_na_specialities_x(df: pd.DataFrame)-> Callable:
    """ replaces NaN with an empty string in 'specialities_x' 
    column
    """
    new = df.copy()
    new['specialities_x'] =  new['specialities_x'].fillna('')
    return apply_remove_emojis(new)

def remove_emojis(text: str)-> str:
    """ removes emojis and :emoji: syntax from text.
    """
    no_emojis_text = emoji.demojize(text)
    cleaned_text = re.sub(r':[^:]+:', '', no_emojis_text)
    return cleaned_text

def apply_remove_emojis(df: pd.DataFrame)-> Callable:
    """ applies method to each row in column 'about'
    """
    new = df.copy()
    new['about'] = new['about'].apply(remove_emojis)
    return apply_convert_to_list(new)

def convert_to_list(text:str)-> List[str]:
    """ Removal of unwanted words and characters from strings.
    Finally, conversion to a list.
    """
    return text.replace('and', '').replace("[", '').replace("]", '').replace("'", '').split(',')

def apply_convert_to_list(df: pd.DataFrame)-> Callable:
    new = df.copy()
    target_columns = ['specialities_x', 'keywords']
    new[target_columns] = new[target_columns].applymap(convert_to_list)
    return merge_columns(new)

def merge_columns(df: pd.DataFrame)-> Callable:
    """ Creates a set of 'keywords' and 'specialities_x' as
    'merged' and drops the two mentioned columns
    """
    new = df.copy()
    new['merged'] = new.apply(lambda row: list(set(row['keywords'] + row['specialities_x'])), axis=1)
    columns_to_drop = ['keywords', 'specialities_x']
    new.drop(columns=columns_to_drop, inplace=True)
    return concat_text(new)

def concat_text(df: pd.DataFrame)-> Callable:
    """ Adds the new column 'merged' into 'about' 
    so there is only one column with text.
    """
    new = df.copy()
    new['merged'] = new['merged'].apply(lambda x: ' '.join(x))
    new['about'] = new['about']  + new['merged']
    new.drop(columns=['merged'], inplace=True)
    return renaming(new)

def renaming(df: pd.DataFrame)-> pd.DataFrame:
    """ Finally cleaning step, renames columns for ease,
    drops any duplicates and returns the final clean df.
    """
    new = df.copy()
    new.rename(columns={'Label': 'label'}, inplace=True)
    new.rename(columns={'about': 'text'}, inplace=True)
    new = new.drop_duplicates()
    return new