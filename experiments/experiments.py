from joblib import load
import time
from game import play
import pandas as pd
import os

FILE = 'chess_results.csv'

datasets = ['master', 'beginner', 'random']
features = [0, 1, 2, 3]
weightings = [0]
opponents = ['random', 'stockfish']
num_games = 30
implementation = 'substitution'


errors = []
all_results = []
initial_time = time.time()

for dataset in datasets:
    for feature in features:
        model = load(f'model_{dataset}_{feature}.joblib')
        scaler = load(f'scaler_{dataset}_{feature}.joblib')
        for nb_weight in weightings:
            for o in opponents:
                for i in range(num_games):
                    try:
                        start = time.time()
                        game_stats = play(model, scaler, o, i, implementation, feature, nb_weight)
                        
                        #save result 
                        end = time.time()
                        print(f"Game {i} took {(end - start)} seconds")

                        for stat in game_stats:
                            stat_copy = stat.copy()
                            stat_copy['dataset'] = dataset
                            stat_copy['feature_selection'] = feature
                            stat_copy['implementation'] = implementation
                            stat_copy['opponent'] = o
                            stat_copy['colour'] = 'white'
                            all_results.append(stat_copy)
                            

                    except Exception as e:
                        errors.append((dataset, feature, o, i, e))
                        continue

                df = pd.DataFrame(all_results)
                if os.path.exists(FILE):
                    existing_df = pd.read_csv(FILE)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_csv(FILE, index=False)
                else:
                    df.to_csv(FILE, index=False)
                print("Results saved")
                all_results = []
    

print("All games played")
t = time.time()
if errors:
    print("Errors:")
    for e in errors:
        print(f"Dataset: '{e[0]}' Feature: '{e[1]}' Opponent: '{e[2]}' Game: '{e[3]}' Error: '{e[4]}'")
print(f"Total time: {((t - initial_time)/60/60)} hours")



datasets = ['master', 'beginner', 'random']
features = [0, 1, 2, 3]
weightings = [0.25, 0.5, 0.75]
opponents = ['random', 'stockfish']
num_games = 30
implementation = 'integration'


errors = []
all_results = []
initial_time = time.time()

for dataset in datasets:
    for feature in features:
        model = load(f'model_{dataset}_{feature}.joblib')
        scaler = load(f'scaler_{dataset}_{feature}.joblib')
        for nb_weight in weightings:
            for o in opponents:
                for i in range(num_games):
                    try:
                        start = time.time()
                        game_stats = play(model, scaler, o, i, implementation, feature, nb_weight)
                        
                        end = time.time()
                        print(f"Game {i} took {(end - start)} seconds")

                        for stat in game_stats:
                            stat_copy = stat.copy()
                            stat_copy['dataset'] = dataset
                            stat_copy['feature_selection'] = feature
                            stat_copy['implementation'] = implementation
                            stat_copy['opponent'] = o
                            stat_copy['colour'] = 'white'
                            all_results.append(stat_copy)
                            

                    except Exception as e:
                        errors.append((dataset, feature, o, i, e))
                        continue

                df = pd.DataFrame(all_results)
                if os.path.exists(FILE):
                    existing_df = pd.read_csv(FILE)
                    combined_df = pd.concat([existing_df, df], ignore_index=True)
                    combined_df.to_csv(FILE, index=False)
                else:
                    df.to_csv(FILE, index=False)
                print("Results saved")
                all_results = []
    

print("All games played")
t = time.time()
if errors:
    print("Errors:")
    for e in errors:
        print(f"Dataset: '{e[0]}' Feature: '{e[1]}' Opponent: '{e[2]}' Game: '{e[3]}' Error: '{e[4]}'")
print(f"Total time: {((t - initial_time)/60/60)} hours")