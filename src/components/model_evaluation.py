import pandas as pd
from src.utilis import load_obj

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns


import os


class model_evaluation:
    def __init__(self,test_data_path='Data/preprocess/test_data.csv',result_dir='src/Results'):
        self.test_data_path = test_data_path
        self.result_dir = result_dir

    def evaluation(self):
        df = pd.read_csv(self.test_data_path)

        model = load_obj('src/models/model.pkl')

        test_data = pd.read_csv(self.test_data_path)
        
        X_test = test_data.drop(['labels'], axis=1)
        y_test = test_data['labels']
        
        y_pred = model.predict(X_test)
                
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        os.makedirs(self.result_dir, exist_ok=True)
        
        # ----- Save Accuracy & Classification Report -----
        metrics_file = os.path.join(self.result_dir, 'metrics.json')
        metrics_data = {
            "accuracy": accuracy,
            "classification_report": report
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)
        print(f"Metrics saved to {metrics_file}")
        
       
        
        print("\nEvaluation Complete!")
        print(f"Accuracy: {accuracy:.4f}")
        

me = model_evaluation()
me.evaluation()