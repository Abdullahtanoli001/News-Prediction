from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from src.utilis import save_obj

import pandas as pd
import numpy as np

import os



class data_transformation:
    def __init__(self,input_path='Data/preprocess/preprocess_data_V1.csv', output_path='Data/preprocess'):
        self.input_path = input_path
        self.output_path = output_path

    def transformation(self):

        df = pd.read_csv(self.input_path)

        numeric_features = ['no_sentences', 'Flesch Reading Ease Score', 'Dale-Chall Readability Score']
        # Filter numeric features that exist in df
        numeric_features = [col for col in numeric_features if col in df.columns]


        

        # Create X and y
        X = df[['combined_text'] + numeric_features]
        y = df['labels']

        # Train-test split
        train_data, test_data= train_test_split(
            df, test_size=0.2, random_state=42
        )
        le = LabelEncoder()
        save_obj('src/models/label_encoder.pkl',le)
        train_data['label_encoded'] = le.fit_transform(train_data['labels'])
        test_data['label_encoded'] = le.transform(test_data['labels'])

        # Preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', TfidfVectorizer(max_features=5000), 'combined_text'),
                ('num', StandardScaler(), numeric_features)
            ]
        )

        save_obj('src/models/preprocessor.pkl',preprocessor)

        os.makedirs(self.output_path,exist_ok=True)
        output_path=os.path.join(self.output_path,'train_data.csv')
        train_data.to_csv(output_path)
        output_path=os.path.join(self.output_path,'test_data.csv')
        test_data.to_csv(output_path)
        output_path=os.path.join(self.output_path,'preprocess_data_V2.csv')
        df.to_csv(output_path)


dt = data_transformation()
dt.transformation()
    
    
