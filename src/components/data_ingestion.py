import pandas as pd
import sys
import os 


class dataingestion:
    def __init__(self,input_path='test_data/bbc_news_text_complexity_summarization.csv',output_path='Data/Raw'):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self):
        try:
            print("Reading Data")
            df = pd.read_csv(self.input_path)
            print("Done Reading Data")

            return df
        except Exception as e:
            raise ValueError(e,sys)
        
    def save_data(self,df):
        try:
            print("Saving Data...")
            os.makedirs(self.output_path,exist_ok=True)
            output_path=os.path.join(self.output_path,'raw_data.csv')
            df.to_csv(output_path)
            print("Data saved")
        except Exception as e:
            raise ValueError(e,sys)
        
di = dataingestion()
df = di.read_data()
di.save_data(df)

