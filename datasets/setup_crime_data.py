import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



class CrimeData:

    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.setup()

    def setup(self):
        df = pd.read_csv('datasets/austin-crime.csv')

        le_location_type = LabelEncoder()
        le_family_violence = LabelEncoder()
        le_zipcode = LabelEncoder()

        df['location_type_n'] = le_location_type.fit_transform(df['Location Type'])
        df['family_violence_n'] = le_family_violence.fit_transform(df['Family Violence'])
        df['zip_code_n'] = le_zipcode.fit_transform(df['Zip Code'])

        loc_mappings = dict(zip(
            le_location_type.classes_, 
            le_location_type.transform(
                le_location_type.classes_
            )
        ))
        zipcode_mappings = dict(zip(
            le_zipcode.classes_,
            le_zipcode.transform(
                le_zipcode.classes_
            )
        ))

        print('Location Type Mappings: ', loc_mappings)
        print('Zip code Mappings: ', zipcode_mappings)

        X = df[['location_type_n', 'zip_code_n']]
        y = df[['family_violence_n']]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    def get_test_data(self):
        return self.x_test, self.y_test
    
    def get_training_data(self):
        return self.x_train, self.y_train
