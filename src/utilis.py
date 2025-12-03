import re
import pickle
import os
import sys


def clean_text(text):
            #lowercase
            text = text.lower()

            #remove urls
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

            #remove digits
            text = re.sub(r'\s+', ' ',text).strip()

            
def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise ValueError(e, sys)
    

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj
            