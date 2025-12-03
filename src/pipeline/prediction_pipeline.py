import pandas as pd 
import numpy as np
import os 
import sklearn

from src.utilis import load_obj


class prediction_pipeline:
    def __init__(self,input_path='src/models/model.pkl'):
        self.input_path = input_path

    def prediction(self,data:dict):

        print('Loading model...')
        model = load_obj('src/models/model.pkl')
        
        df = pd.DataFrame([data])

        df['combined_text'] = (
            df['text'] + " " +
            df['text_rank_summary'] + " " +
            df['lsa_summary']
        )

        pred = model.predict(df)
        print('Pridiction:', pred)

data = {

            'text':'Honda wins China copyright',
            'text_rank_summary': 'And on Tuesday, Paws Incorporated - the owner',
            'lsa_summary': 'Internationally recognized regulation is now a',
        }

pp = prediction_pipeline()
pp.prediction(data)
