from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


from src.components.data_transformation import data_transformation
from src.utilis import load_obj, save_obj

import pandas as pd 



class model_training:
    def __init__(self,test_data_path='Data/preprocess/test_data.csv', train_data_path='Data/preprocess/train_data.csv',output_save_path='src/models'):
        self.test_data_path=test_data_path
        self.train_data_path = train_data_path
        self.output_save_path = output_save_path


    def training(self):
        print("Reading Data")
        train_data = pd.read_csv(self.train_data_path)
        test_data = pd.read_csv(self.test_data_path)

        print("Loading data")
        preprocessor = load_obj('src/models/preprocessor.pkl')


        X_train = train_data.drop(['labels'],axis=1)
        y_train = train_data['labels']

        X_test = test_data.drop(['labels'],axis=1)
        y_test = test_data['labels']


        print("Creating Pipeline")
        pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(max_depth= 20, n_estimators=200))
        ])
        print("Creating model...")
        # Train
        model = pipeline.fit(X_train, y_train)

        save_obj('src/models/model.pkl',model)

mt = model_training()
mt.training()



        

        