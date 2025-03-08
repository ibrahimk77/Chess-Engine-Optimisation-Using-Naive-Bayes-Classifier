import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score
from joblib import dump
from naive_bayes import NaiveBayes



def prepare_data(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, :-1])
    y = data.iloc[:, -1]
    return X, y, scaler

def train():
    data = pd.read_csv('chess_games_features.csv')

    X, y, scaler = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    model = NaiveBayes()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    for c in np.unique(y_train):
        print(f"Class {c}: {np.sum(y_train == c)}")

    matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Confusion Matrix: ")
    print(matrix)
    print("F1 Score: ", f1)


    dump(model, 'model.joblib')
    dump(scaler, 'scaler.joblib')

    print("Model saved.")




train()
