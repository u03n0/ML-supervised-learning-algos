import unittest
from src.utils import save_data
import pandas as pd



class TestSavingData(unittest.TestCase):

    def test_csv_extention(self):
        """ Tests that csv extention is 
        being used.
        """
        df = pd.DataFrame()
        filename = 'dummy_file_for_testing.csv'
        result = save_data(df, filename=filename)
        self.assertEqual(result, None)


    def test_not_csv_extention(self):
        """Tests that ValueError is raised
        if other than .csv extention is entered.
        """
        df = pd.DataFrame()
        filename = 'dummy_file_for_testing.txt'
        with self.assertRaises(ValueError):
            save_data(df, filename=filename)