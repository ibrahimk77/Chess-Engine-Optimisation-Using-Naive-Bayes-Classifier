import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score

class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.stds = {}
        self.means = {}
        self.classes = None
        self.epsilon = 1e-10
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        
        # Calculate priors
        for c in self.classes:
            self.priors[c] = np.sum(y == c) / len(y)
            
        # Calculate means and stds for each feature for each class
        for c in self.classes:
            class_data = X[y == c]
            self.means[c] = np.mean(class_data, axis=0)
            self.stds[c] = np.std(class_data, axis=0) + self.epsilon
                
    def _calculate_likelihood(self, x, mean, std):
        exponent = -0.5 * ((x - mean) / std) ** 2
        return np.exp(exponent) / (std * np.sqrt(2 * np.pi))
    
    def predict(self, X):
        predictions = []
        
        for x in X:
            # Calculate posterior for each class
            posteriors = {}
            
            for c in self.classes:
                # Start with prior
                posterior = np.log(self.priors[c])
                
                # Add log likelihoods
                for i, feature_val in enumerate(x):
                    likelihood = self._calculate_likelihood(
                        feature_val, 
                        self.means[c][i], 
                        self.stds[c][i]
                    )
                    # Add small constant to prevent log(0)
                    posterior += np.log(likelihood + self.epsilon)
                
                posteriors[c] = posterior
            
            # Select class with highest posterior
            predictions.append(max(posteriors.items(), key=lambda x: x[1])[0])
            
        return np.array(predictions)

# Data preprocessing
def prepare_data(data):
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:, :-1])
    y = data.iloc[:, -1].values
    return X, y

# Training and evaluation
def train_and_evaluate():
    # Load data
    data = pd.read_csv('chess_games_features.csv')
    
    # Prepare data
    X, y = prepare_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = NaiveBayes()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return conf_matrix, f1, model

if __name__ == "__main__":
    conf_matrix, f1, model = train_and_evaluate()
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nF1 Score:", f1)