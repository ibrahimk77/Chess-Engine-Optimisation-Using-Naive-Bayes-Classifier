import pandas as pd
import chess 
import re
from features import board_features
import time


def an_to_moves(an):
    moves = re.sub(r"\d+\.", "", an)
    moves = re.sub(r"1-0|0-1|1/2-1/2", "", moves)
    moves = moves.strip()
    moves_list = moves.split()
    return moves_list

def features_from_AN(an, result, model_num):

    moves = an_to_moves(an)
    board = chess.Board()
    features_at_intervals = []


    total_moves = len(moves)
    end_game_start = (total_moves//4) * 3

    if result == '1-0':
        result = 1
    elif result == '0-1':
        result = -1
    else:
        result = 0
    for i, move in enumerate(moves):

        try:
            board.push_san(move)
        except ValueError:
            break

        extract = False

        if i < end_game_start and i % 6 == 0:
            extract = True
        
        elif i % 2 == 0:
            extract = True
        
        if i == total_moves - 1:
            extract = True
    
        if extract: # or i == len(moves) - 1:
            
            features = board_features(board, model_num)
            features['result'] = result
            features_at_intervals.append(features)
    
    return features_at_intervals



def preprocess_data(input_file, output_file, i):

    df = pd.read_csv(input_file)

    all_features = []
    
    total_time = 0
    values = 0

    for index, row in df.iterrows():
        start = time.time()
        if row['Result'] == '1/2-1/2': #TODO:Remove draws for now 
            continue
        features = features_from_AN(row['AN'], row['Result'], i)
        all_features.extend(features)
        end = time.time()
        total_time += end - start
        values += 1
    
    new_df = pd.DataFrame(all_features)
    new_df.to_csv(output_file, index=False)
    print(f"Average time taken for each move: {total_time/values}")

def preprocess_data_4():
    for i in range(4):
        preprocess_data(f'chess_games_100k.csv', f'chess_games_features_{i}.csv', i)
        
#preprocess_data('chess_games_10k.csv', 'chess_games_features.csv', 1)
preprocess_data_4()



