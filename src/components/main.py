from src.components.data_ingestion import dataingestion
from src.components.data_preprocessing import preprocessing
from src.components.data_transformation import data_transformation
from src.components.model_training import model_training

#data ingestion
di = dataingestion()
df = di.read_data()
di.save_data(df)


#data preprocessing
pp = preprocessing()
pp.preprocess()


#data transformation
dt = data_transformation()
dt.transformation()

#data training
mt = model_training()
mt.training()
