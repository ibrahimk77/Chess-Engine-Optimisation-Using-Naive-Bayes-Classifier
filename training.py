import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
sns.set_style('darkgrid')


def calculate_prior(df,Y):
    classes = sorted(list(df[Y].unique())) # Unique classes 
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df)) # P(C)
    return prior

def calculate_likelihood_categorical(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    p_x_given_y = len(df[df[feat_name]==feat_val]) / len(df)  # Likelihood P(X|C)
    return p_x_given_y


def naive_bayes_categorical(df, X, Y):

    features = list(df.columns)[:-1]

    prior = calculate_prior(df, Y)

    Y_pred = []

    for x in X:
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(df, features[i], x[i], Y, labels[j])

        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j] # P(C|X)

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred)



def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    return p_x_given_y


def naive_bayes_guassian(df, X, Y):

    features = list(df.columns)[:-1]

    prior = calculate_prior(df, Y)

    Y_pred = []

    for x in X:
        labels = sorted(list(df[Y].unique()))
        likelihood = [1] * len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])
        
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]
        
        Y_pred.append(labels[np.argmax(post_prob)])

    return np.array(Y_pred)



data = pd.read_csv('chess_games_features.csv')

# category_mapping = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}
# data['result_encoded'] = data['result'].map(category_mapping)


# data = data[["material_balance", "result_encoded"]]



train, test = train_test_split(data, test_size=0.2)

X_test = test.iloc[:,:-1].values
Y_test = test.iloc[:,-1].values
Y_pred = naive_bayes_guassian(train, X=X_test, Y='result')

print(confusion_matrix(Y_test, Y_pred))
print('F1 Score: ', f1_score(Y_test, Y_pred, average='weighted'))

# Check the distribution of the 'result' column in the training and test sets
print(train['result'].value_counts())
print(test['result'].value_counts())





































# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score, classification_report

# df = pd.read_csv('chess_games_small_features.csv')
# X = df['material_balance']
# Y = df['result']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# classifier = GaussianNB()
# classifier.fit(X_train, y_train)

# # Make predictions
# y_pred = classifier.predict(X_test)

# # Evaluate the model
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
