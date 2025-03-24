import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, accuracy_score, recall_score, precision_score
from joblib import dump
from naive_bayes import NaiveBayes



def prepare_data(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, :-1])
    y = data.iloc[:, -1]
    return X, y, scaler

def train(dataset, model_num, data):
    X, y, scaler = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    model = NaiveBayes()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(np.unique(y_pred))    
    print(np.unique(y_test))
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    eval = {
        'Dataset': dataset,
        'model': model_num,
        'f1': f1,
        'kappa': kappa,
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision
        }
    

    dump(model, f'model_{dataset}_{model_num}.joblib')
    dump(scaler, f'scaler_{dataset}_{model_num}.joblib')

    print(f"Model {dataset} {model_num} saved.")
    return eval

def train_multiple():
    for i in range(4):
        data = pd.read_csv(f'chess_games_features_{i}.csv')

        X, y, scaler = prepare_data(data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

        model = NaiveBayes()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        for c in np.unique(y_train):
            print(f"Class {c}: {np.sum(y_train == c)}")

        matrix = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print("Confusion Matrix: ")
        print(matrix)
        print("F1 Score: ", f1)
        print("Kappa Score: ", kappa)

        print("Accuracy: ", accuracy)
        print("Recall: ", recall)
        print("Precision: ", precision)


        dump(model, f'model{i}.joblib')
        dump(scaler, f'scaler{i}.joblib')

        print(f"Model{i} saved.")




#train_multiple()
