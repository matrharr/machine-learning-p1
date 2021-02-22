import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


class DataLoader:

    def __init__(self, name):
        self.data_name = name

    def load_data(self):
        if self.data_name == 'bankrupt':
            # return self._load_loan_data()
            return self._load_bankrupt_data()
        else:
            # return self._load_video_game_data()
            return self._load_brain_tumor_data()
    
    def _load_brain_tumor_data(self):
        brain = pd.read_csv('datasets/brain-tumor.csv')
        # remove missing values
        # feature scaling - are numerical attributes at very diff scales?
        brain.drop(['Image'], axis=1, inplace=True)
        ohe = OneHotEncoder()
        col_trans = make_column_transformer(
            # ('passthrough', [
            #     'Debt ratio',
            #     'Current Liability to Assets',
            #     'Borrowing dependency',
            #     'Current Liability to Current Assets'
            # ]),
            (ohe, []),
            remainder='passthrough'
        )
        return col_trans, brain


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

    def _load_video_game_data(self):
        videogame = pd.read_csv('datasets/video-game-rating.csv')
        videogame.drop(['title'], axis=1, inplace=True)
        [print(col) for col in videogame.columns]
        ohe = OneHotEncoder()
        col_trans = make_column_transformer(
            (ohe, ['']),
            remainder='passthrough'
        )
        return col_trans, videogame

    def _load_bankrupt_data(self):
        bankrupt = pd.read_csv('datasets/company-bankrupt.csv')
        # remove missing values
        # if needed, convert text values to categorical (one hot)
        # feature scaling - are numerical attributes at very diff scales?
        ohe = OneHotEncoder()
        col_trans = make_column_transformer(
            # ('passthrough', [
            #     'Debt ratio',
            #     'Current Liability to Assets',
            #     'Borrowing dependency',
            #     'Current Liability to Current Assets'
            # ]),
            (ohe, []),
            remainder='passthrough'
        )
        return col_trans, bankrupt
