import chess
from predict import predict_new_data_prob

MODEL_NUM = 3


def alphaBeta(board, alpha, beta, depth, is_maximising):

    if depth == 0 or board.is_game_over():
        return (evaluate(board), None)

    legal_moves = list(board.legal_moves)

    best_move = legal_moves[0]

    legal_moves = sorted(legal_moves, key=lambda move: evaluate_move(move, board), reverse=True)

    if is_maximising:
        val = -float("inf")
        for move in legal_moves:
            board.push(move)
            current_val = alphaBeta(board, alpha, beta, depth - 1, False)[0]
            board.pop()

            if current_val > val:
                val = current_val
                best_move = move
            
            alpha = max(alpha, current_val)
            if beta <= alpha:
                break

        return (val, best_move)

    else:
        val = float("inf")
        for move in legal_moves:
            board.push(move)
            current_val = alphaBeta(board, alpha, beta, depth - 1, True)[0]
            board.pop()

            if current_val < val:
                val = current_val
                best_move = move
            
            beta = min(beta, current_val)
            if beta <= alpha:
                break

        return (val, best_move)

def evaluate(board):
        
    prediction = predict_new_data_prob(board, MODEL_NUM)

    score = prediction.get(1,0) - prediction.get(-1,0)
    evaluation = score


    if board.is_checkmate():
        if board.turn == chess.WHITE:
            evaluation = 999999999999
        else:
            evaluation = -999999999999

    return evaluation

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
