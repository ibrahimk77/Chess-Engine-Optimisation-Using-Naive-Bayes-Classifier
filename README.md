User Guide
There are two main components of the codebase:

Training the Model

Playing Chess Games (Experiments)

Training the Model
Before training, please follow these steps:

Download the Dataset:
Download the Kaggle dataset from this link and rename the file to chess_games.csv.

Install Required Libraries:
Install all required libraries as listed in the requirements.txt file.

The main files required for training are:

dataset_prep.py
This is the main file to execute to train the 12 models. It loads the data and separates it into three groups (master level games, beginner level games, and a random sample of games). For each combination of dataset and feature set, it preprocesses the data by invoking the preprocess_data function from data_prep.py, trains the model using the train function from training.py, and saves the evaluation metrics to a CSV file named eval_results.csv.

data_prep.py
This file is used to preprocess the data and prepare it for training. It simulates each game in the dataset and extracts features at set intervals. The processed data returned by the preprocess_data function is then used to train the model.

features.py
Contains all the methods for feature extraction. Features are extracted based on the selected feature set.

naive_bayes.py
This is the main file for the Naive Bayes model. It contains two primary methods:

predict: Returns the predicted class for a given set of features.

predict_prob: Returns the probabilities for each class.

training.py
This file handles the main training process. It first scales the data using the prepare_data function and then splits it into training and testing sets. It trains the model by calling the fit method from naive_bayes.py and then calculates various metrics (such as accuracy, precision, recall, kappa, and F1 score). Finally, it saves the models and scalers as joblib files.

Playing Chess Games (Experiments)
Before running the experiments:

Install Stockfish:
Install Stockfish ideally in the same directory as the game.py file. If it is not in the same directory, update the STOCKFISH_PATH variable in game.py to point to the correct location of the Stockfish executable. Stockfish can be downloaded from here.

The main files required for running experiments are:

experiments.py
This is the main file to execute to run the experiments. It iterates over all combinations of datasets, feature sets, Naive Bayes weightings, opponents, and implementations, playing 30 games for each combination. It loads the models and scalers from the joblib files, calls the play function from game.py to run each game, and saves the results to a CSV file named game_results.csv.

game.py
This file contains the code for playing chess games. The play function tracks various metrics, creates a new chess board, and selects moves using the engine based on the get_alphaBeta_move function (which depends on the selected implementation). Moves can be chosen either via a random engine or Stockfish (depending on the chosen opponent). The game continues until completion, and the resulting metrics are saved in a CSV file.

minimax.py
Contains the alphaBeta function, which implements a standard minimax algorithm with alpha-beta pruning.

minimax_NB_integrated.py
Contains the alphaBeta_integrated function, which implements the MMNB integration algorithm.

minimax_NB_sub.py
Contains the alphaBeta_sub function, which implements the MMNB substitution algorithm.