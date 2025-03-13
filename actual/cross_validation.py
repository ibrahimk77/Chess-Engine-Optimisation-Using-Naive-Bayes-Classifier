from naive_bayes import NaiveBayes
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from training import prepare_data
import pandas as pd
import numpy as np
#TODO: Mention in report using kfold to find best epsilon
#TODO: CHANGE CODE

epsilons = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000] #TODO: TREY OTHER values like 0.5, 0.75
best_epsilon = None
best_score = 0

kf = KFold(n_splits=5, shuffle=True, random_state=42)
data = pd.read_csv('chess_games_features.csv').head(500000)
X, y, scaler = prepare_data(data)

for eps in epsilons:
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = NaiveBayes()
        model.set_epsilon(eps)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = f1_score(y_test, preds, average='macro')  # Adjust average as needed
        scores.append(score)
    avg_score = np.mean(scores)
    print(f"Epsilon: {eps}, Average F1 Score: {avg_score:.4f}")
    if avg_score > best_score:
        best_score = avg_score
        best_epsilon = eps

print(f"Best epsilon: {best_epsilon} with F1 Score: {best_score:.4f}")