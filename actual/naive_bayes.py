import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.stds = {}
        self.means = {}
        self.classes = None
        self.epsilon = 1e-10 #TODO: CHANGE
    
    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            self.priors[c] = np.sum(y == c) / len(y)

        for c in self.classes:
            class_data = X[y == c]
            self.means[c] = np.mean(class_data, axis=0)
            self.stds[c] = np.std(class_data, axis=0) + self.epsilon
    
    def calculate_likelihood(self, x, mean, std):
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def predict(self, X):
        predictions = []

        for x in X:
            posteriors = {}

            for c in self.classes:

                posterior = np.log(self.priors[c]) #TODO: Mention log in report 

                for i, feature_val in enumerate(x):
                    likelihood = self.calculate_likelihood(
                        feature_val, 
                        self.means[c][i], 
                        self.stds[c][i]
                    )

                    posterior += np.log(likelihood + self.epsilon) #TODO: Mention add small constant to prevent log(0) in report

                posteriors[c] = posterior

            predictions.append(max(posteriors.items(), key=lambda x: x[1])[0]) # selects highest posterior
        
        return np.array(predictions)

