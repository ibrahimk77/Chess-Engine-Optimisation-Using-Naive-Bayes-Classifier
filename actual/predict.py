import numpy as np
import pandas as pd
from joblib import load
from naive_bayes import NaiveBayes

#TODO: FINISH

def load_model_and_scaler():
    model = load('model.joblib')
    scaler = load('scaler.joblib')
    return model, scaler

def predict_new_data():
    model, scaler = load_model_and_scaler()
    X = scaler.transform(data)
    return model.predict(X





