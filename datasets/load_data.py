import pandas as pd


class DataLoader:

    def __init__(self, name):
        self.data_name = name

    def load_data(self):
        if self.data_name == 'loan':
            return self._load_loan_data()
        else:
            return self._load_bankrupt_data()
    
    def _load_loan_data(self):
        loan = pd.read_csv('loan-default.csv')
        # use pipeline for below?
        # remove missing values
        # if needed, convert text values to categorical (one hot)
        # feature scaling - are numerical attributes at very diff scales?

        return loan

    def _load_bankrupt_data(self):
        bankrupt = pd.read_csv('datasets/company-bankrupt.csv')
        # use pipeline for below?
        # remove missing values
        # if needed, convert text values to categorical (one hot)
        # feature scaling - are numerical attributes at very diff scales?

        return bankrupt
