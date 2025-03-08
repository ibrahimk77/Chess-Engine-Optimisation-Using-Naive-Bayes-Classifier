import pandas as pd
import chess 
import re
from features import calculate_material_balance, piece_mobility, control_of_center_small, control_of_center_large, king_attack_balance




def an_to_moves(an):
    moves = re.sub(r"\d+\.", "", an)
    moves = re.sub(r"1-0|0-1|1/2-1/2", "", moves)
    moves = moves.strip()
    moves_list = moves.split()
    return moves_list


def extract_features_from_AN(an, result):

    moves = an_to_moves(an)
    board = chess.Board()
    features_at_intervals = []

    mobility = 0




    for i, move in enumerate(moves):

        try:
            board.push_san(move)
        except ValueError:
            break

        
        if result == '1-0':
            result = 1
        elif result == '0-1':
            result = -1
        else:
            result = 0


        if i > len(moves)/2 and i%9==0 and board.turn == chess.WHITE: # or i == len(moves) - 1:
            material_balance = calculate_material_balance(board)
            #if material_balance != 0 : # TODO/REMOVE: just for now

            mobility = piece_mobility(board)
            centre_small = control_of_center_small(board)
            centre_large = control_of_center_large(board)
            king_attack = king_attack_balance(board)


            features_at_intervals.append(
                {
                    #'fen': board.fen(),
                    'result': result,
                    'material_balance': material_balance,
                    'piece_mobility': mobility,
                    'control_of_center_small': centre_small,
                    'control_of_center_large': centre_large,
                    'king_attack_balance': king_attack
                }
            )

    return features_at_intervals


df = pd.read_csv('chess_games_small.csv')

#df = pd.read_csv('chess_games.csv').head(10000)



all_features = []

for index, row in df.iterrows():

    
    features = extract_features_from_AN(row['AN'], row['Result'])
    all_features.extend(features)

new_df = pd.DataFrame(all_features)
new_df.to_csv('chess_games_features.csv', index=False) 



    



