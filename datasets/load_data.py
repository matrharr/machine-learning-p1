import pandas as pd

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
        # use pipeline for below?
        # remove missing values
        # if needed, convert text values to categorical (one hot)
        # feature scaling - are numerical attributes at very diff scales?
        employment_type = OneHotEncoder()
        employment_type.fit_transform(loan[['Employment.Type']])

        cns_score_description = OneHotEncoder()
        cns_score_description.fit_transform(loan[['PERFORM_CNS.SCORE.DESCRIPTION']])
        
        loan = loan.drop([
            'DisbursalDate',
            'Date.of.Birth',
            'Employment.Type',
            'PERFORM_CNS.SCORE.DESCRIPTION',
            'AVERAGE.ACCT.AGE',
            'CREDIT.HISTORY.LENGTH'
            ], axis=1)
        loan['employment_type'] = employment_type
        [print(col) for col in loan.columns]
        return loan

    def _load_bankrupt_data(self):
        bankrupt = pd.read_csv('datasets/company-bankrupt.csv')
        # use pipeline for below?
        # remove missing values
        # if needed, convert text values to categorical (one hot)
        # feature scaling - are numerical attributes at very diff scales?

        return bankrupt
