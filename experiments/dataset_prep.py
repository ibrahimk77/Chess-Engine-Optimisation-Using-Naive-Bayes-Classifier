from data_prep import preprocess_data
from training import train 
import pandas as pd
import numpy as np






games = pd.read_csv('chess_games.csv')
print("Games loaded")
games = games[(games['Result'] == '1-0') | (games['Result'] == '0-1')]




master_games = games[(games['WhiteElo'] > 2200) | (games['BlackElo'] > 2200)]
beginner_games = games[(games['WhiteElo'] < 2200) & (games['BlackElo'] < 2200)].head(len(master_games))
random_games = games.sample(len(master_games))
print("Training: ", len(master_games), " games")


results = []
print("Training models")

for i in range(4):

    master_features = preprocess_data(master_games, 'master_games', i)

    master_eval = train('master', i, master_features)
    print(f"Master model {i} trained")


    beginner_features = preprocess_data(beginner_games, 'beginner_games', i)
    beginner_eval = train('beginner', i, beginner_features)
    print(f"Beginner model {i} trained")


    random_features = preprocess_data(random_games, 'random_games', i)
    random_eval = train('random', i, random_features)
    print(f"Random model {i} trained")

    results.append(master_eval)
    results.append(beginner_eval)
    results.append(random_eval)

eval_df = pd.DataFrame(results)
eval_df.to_csv('eval_results.csv', index=False)
print("Results saved")












