# User Guide

There are two main components of the codebase: one regarding the training of the model and one regarding the playing of chess games.

Before training, it is important to download the kaggle dataset from the link [https://www.kaggle.com/datasets/arevel/chess-games](https://www.kaggle.com/datasets/arevel/chess-games) and rename it to `chess_games.csv`. It is also important to install all required libraries as listed in the `requirements.txt` file. The main files required for the training of the models are:

- **`dataset_prep.py`**: this is the main file to be executed to train the 12 models. This file loads the data, separates the data into three groups, master level games, beginner level games and a random sample of games. For each combination of dataset and feature set, it preprocesses the data by invoking the `preprocess_data` function from `data_prep.py`. It then uses the processed data to train the model using the `train` function from `training.py`. It then saves the evaluation metrics to a CSV file named `eval_results.csv`.

- **`data_prep.py`**: this file is used to preprocess the data and prepare it for training. It goes through each game in the dataset, simulates the games and extracts features at certain intervals. The `preprocess_data` function returns the processed data, which is then used to train the model.

- **`features.py`**: contains all the methods to extract each feature. It also extracts features based on the feature set selected.

- **`naive_bayes.py`**: This is the main file for the Naive Bayes model. It contains two main methods `predict` which returns the predicted class for a given set of features and `predict_prob` which returns the probabilities for each class.

- **`training.py`**: This file is where the main training occurs. The data is first scaled using the `prepare_data` function. It then splits the data into training and testing data. It invokes the `fit` method from the `naive_bayes.py` file to train the model. It then calculates a number of metrics such as accuracy, precision, recall, kappa and F1 score and returns these metrics. It also saves the models and scalers to joblib files.

Before running the experiments, it is important to install Stockfish ideally in the same directory as the `game.py` file. If it is not downloaded in the same directory, make sure to alter the `STOCKFISH_PATH` in the `game.py` file to the path of the Stockfish executable. It can be downloaded from the link [https://stockfishchess.org/download/](https://stockfishchess.org/download/).

The main files required for the running of the experiments are:

- **`experiments.py`**: this is the main file to be executed to run the experiments. It goes through all combinations of datasets, feature sets, Naive Bayes weightings, opponents and implementations and plays 30 games for each combination. It loads the models from the joblib files as well as the respective scalers. It calls the `play` function from the `game.py` file to play the games. It then saves the results to a CSV file named `game_results.csv`.

- **`game.py`**: this is the file where the game is played. In the `play` function, it keeps track of a number of metrics. It creates a new board and chooses a move by the engine based on the `get_alphaBeta_move` function, depending on the implementation selected. It then chooses a move either by a random engine or Stockfish, depending on the opponent selected. It then continues this until the game is over. It then returns the different metrics of the game to be saved in a CSV file.

- **`minimax.py`**: this file contains the `alphaBeta` function which implements a normal minimax algorithm with alpha-beta pruning.

- **`minimax_NB_integrated.py`**: this file contains the `alphaBeta_integrated` function which includes the implementation of the MMNB integration algorithm.

- **`minimax_NB_sub.py`**: this file contains the `alphaBeta_sub` function which includes the implementation of the MMNB substitution algorithm.
