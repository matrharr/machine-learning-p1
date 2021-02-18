import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


class DataLoader:

    def __init__(self, name):
        self.data_name = name

    def load_data(self):
        if self.data_name == 'loan':
            return self._load_loan_data()
        else:
            return self._load_bankrupt_data()
    
    def _load_loan_data(self):
        loan = pd.read_csv('datasets/loan-default.csv')
        # remove missing values
        # if needed, convert text values to categorical (one hot)
        # feature scaling - are numerical attributes at very diff scales?
        ohe = OneHotEncoder()
        [print(col) for col in loan.columns]

        col_trans = make_column_transformer(
            (ohe, ['Employment.Type', 'PERFORM_CNS.SCORE.DESCRIPTION']),
            ('drop', [
                'DisbursalDate',
                'Date.of.Birth',
                'AVERAGE.ACCT.AGE',
                'CREDIT.HISTORY.LENGTH'
            ]),
            remainder='passthrough'
        )
        return col_trans, loan

    def _load_bankrupt_data(self):
        bankrupt = pd.read_csv('datasets/company-bankrupt.csv')
        # use pipeline for below?
        # remove missing values
        # if needed, convert text values to categorical (one hot)
        # feature scaling - are numerical attributes at very diff scales?
        ohe = OneHotEncoder()
        col_trans = make_column_transformer(
            (ohe, []),
            remainder='passthrough'
        )
        return col_trans, bankrupt
