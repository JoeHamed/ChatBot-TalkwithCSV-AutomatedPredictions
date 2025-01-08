import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

class ChurnPreProcessing:
    @staticmethod
    def preprocess(data):

        # Drop customerID column (isn't useful in classification)
        if 'customerID' in data.columns:
            data.drop('customerID', axis='columns', inplace=True)

        # print(data.info())

        # Total_charges column should be of numerical type
        data['Total_Charges'] = data['Total_Charges'].apply(pd.to_numeric,
                                                            errors='coerce')  # invalid parsing will be set as NaN

        # print(data.isnull().sum())

        # Drop rows with NaN in the 'Total_Charges' column
        data = data.dropna(subset=['Total_Charges'])

        # print(data.isnull().sum())

        # Drop duplicates
        data.drop_duplicates(inplace=True)

        # Checking unique values to choose which technique to apply
        should_be_one_hot_encoded = []
        should_be_label_encoded = []

        # Make a copy of the DataFrame
        data = data.copy()

        # Handling hidden redundancy
        data.replace('No internet service', 'No', inplace=True)
        data.replace('No phone service', 'No', inplace=True)


        for col in data.columns:
            if data[col].dtypes == 'object':  # Exclude numerical values
                if len(data[col].unique()) > 2: # has more than 2 unique values
                    should_be_one_hot_encoded.append(col)
                else:
                    should_be_label_encoded.append(col)


        # Label Encoding
        le = LabelEncoder()
        # print(should_be_label_encoded)
        for col in should_be_label_encoded:
            data[col] = le.fit_transform(data[col])  # Apply label encoding for each column

        # for col in should_be_label_encoded:
        #     print(f'{col}: {data[col].unique()}')

        # One-Hot Encoding
        data_dumm = pd.get_dummies(data=data, columns=should_be_one_hot_encoded)

        # print(should_be_one_hot_encoded)
        # print(data_dumm.columns)
        # print(data_dumm.info())

        # Feature Scaling
        cols_to_scale = ['tenure', 'Monthly_Charges', 'Total_Charges']
        scaler = MinMaxScaler()
        data_dumm[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

        # Splitting The data into training and test sets
        X = data_dumm.drop('Churn', axis='columns')
        y = data_dumm['Churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Perform over-under sampling
        smote = SMOTEENN(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTEENN, X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        return X_train, X_test, y_train, y_test