import numpy as np
import pandas as pd
from joblib import load
from naive_bayes import NaiveBayes
from sklearn.preprocessing import StandardScaler
from features import board_features
import time

#TODO: FINISH

def load_model_and_scaler(model_num):
    model = load(f'model{model_num}.joblib')
    scaler = load(f'scaler{model_num}.joblib')
    return model, scaler

def predict_new_data(board, i):
    model, scaler = load_model_and_scaler(i)
    features = board_features(board, i)
    data = pd.DataFrame([features])

    X = scaler.transform(data)

    return model.predict(X)[0]

def predict_new_data_prob(board, i):
    model, scaler = load_model_and_scaler(i)
    features = board_features(board, i)
    data = pd.DataFrame([features])

    X = scaler.transform(data)

    return model.predict_prob(X)[0]




