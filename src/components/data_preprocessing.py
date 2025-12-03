from src.utilis import clean_text

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import re
import os 
import sys

class preprocessing:
    def __init__(self, input_path='Data/Raw/raw_data.csv', output_path='Data/preprocess'):
        self.input_path = input_path
        self.output_path = output_path

    def preprocess(self):
        try:
            df = pd.read_csv(self.input_path)
            df = df[['text', 'text_rank_summary', 'lsa_summary', 'labels']]
            df['combined_text'] = (
            df['text'] + " " +
            df['text_rank_summary'] + " " +
            df['lsa_summary']
                )
            df['clean_text'] = df['combined_text'].apply(clean_text)
            os.makedirs(self.output_path,exist_ok=True)
            output_path=os.path.join(self.output_path,'preprocess_data_V1.csv')
            df.to_csv(output_path)
            print("Data Saved")
        except Exception as e:
            raise ValueError(e,sys)


pp = preprocessing()
pp.preprocess()

