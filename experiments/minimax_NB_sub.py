import chess
from naive_bayes import NaiveBayes
from features import board_features
import pandas as pd
import time


def alphaBeta_sub(board, alpha, beta, depth, is_maximising, model, scaler, feature):
    nodes_explored = 1  
    if depth == 0 or board.is_game_over():
        return evaluate(board, model, scaler, feature), None, nodes_explored

    legal_moves = list(board.legal_moves)
    legal = sorted(legal_moves, key=lambda move: evaluate_move(move, board), reverse=True)


    best_move = legal[0]
    if is_maximising:
        value = -float("inf")
        for move in legal:
            board.push(move)
            eval_val, _, nodes_child = alphaBeta_sub(board, alpha, beta, depth - 1, False, model, scaler, feature)
            nodes_explored += nodes_child 
            board.pop()
            if eval_val > value:
                value = eval_val
                best_move = move
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value, best_move, nodes_explored
    else:
        value = float("inf")
        for move in legal:
            board.push(move)
            eval_val, _, nodes_child = alphaBeta_sub(board, alpha, beta, depth - 1, True, model, scaler, feature)
            nodes_explored += nodes_child 
            board.pop()
            if eval_val < value:
                value = eval_val
                best_move = move
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move, nodes_explored


def evaluate(board, model, scaler, feature):
    prediction = predict_new_data_prob(board, model, scaler, feature)
    score = prediction.get(1,0)
    evaluation = score

    if board.is_checkmate():
        if board.turn == chess.WHITE:
            evaluation = -float("inf")
        else:
            evaluation = float("inf")
    
    return evaluation


def predict_new_data_prob(board, model, scaler, i):

    features = board_features(board, i)
    data = pd.DataFrame([features])
    X = scaler.transform(data)

    return model.predict_prob(X)[0]

    
    

def evaluate_move(move, board):

    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }

    target_piece = board.piece_at(move.to_square)
    if target_piece:
        return piece_values[target_piece.piece_type]
    else:
        return 0
