import chess
from predict import predict_new_data_prob


NB_WEIGHT = 0
EVALUATION_WEIGHT = 1
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
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    #TODO: QUOTE THIS
    #https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=00307ba02e3fa2e23eac5e3c35dabeb054054fe3


    PAWN_POSITION_TABLE = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 5, -20, -20, -20, -20, 5, 5],
        [5, 5, 5, 0, 0, 5, 5, 5],
        [0, 0, 0, 5, 5, 0, 0, 0],
        [5, 5, 5, 10, 10, 5, 5, 5],
        [10, 10, 10, 20, 20, 10, 10, 10],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]

    KNIGHT_POSITION_TABLE = [
        [-50, -40, -30, -30, -30, -30, -40, -50],
        [-40, -20, 0, 5, 5, 0, -20, -40],
        [-30, 0, 10, 15, 15, 10, 0, -30],
        [-30, 5, 15, 20, 20, 15, 5, -30],
        [-30, 0, 15, 20, 20, 15, 0, -30],
        [-30, 5, 10, 15, 15, 10, 5, -30],
        [-40, -20, 0, 5, 5, 0, -20, -40],
        [-50, -40, -30, -30, -30, -30, -40, -50]
    ]

    BISHOP_POSITION_TABLE = [
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-10, 0, 5, 5, 5, 5, 0, -10],
        [-10, 5, 10, 10, 10, 10, 5, -10],
        [-10, 5, 10, 15, 15, 10, 5, -10],
        [-10, 5, 10, 15, 15, 10, 5, -10],
        [-10, 0, 5, 5, 5, 5, 0, -10],
        [-20, -10, -10, -10, -10, -10, -10, -20],
        [-20, -10, -10, -10, -10, -10, -10, -20]
    ]

    ROOK_POSITION_TABLE = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [5, 10, 10, 10, 10, 10, 10, 5],
        [5, 10, 15, 15, 15, 15, 10, 5],
        [5, 10, 15, 20, 20, 15, 10, 5],
        [5, 10, 15, 20, 20, 15, 10, 5],
        [5, 10, 15, 15, 15, 15, 10, 5],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]


    KING_POSITION_TABLE = [
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-30, -40, -40, -50, -50, -40, -40, -30],
        [-20, -30, -30, -40, -40, -30, -30, -20],
        [-10, -20, -20, -30, -30, -20, -20, -10],
        [10, 10, 0, 0, 0, 0, 10, 10],
        [20, 20, 10, 10, 10, 10, 20, 20]
    ]


    evaluation = 0

    if board.is_checkmate():
        if board.turn == chess.WHITE:
            return -float("inf")
        else:
            return float("inf")

    if board.is_stalemate():
        return 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            piece_type = piece.piece_type


            if piece.color == chess.WHITE:
                evaluation += piece_values[piece.piece_type]
            else:
                evaluation -= piece_values[piece.piece_type]

            
            file = chess.square_file(square)
            rank = chess.square_rank(square)

            if piece.color == chess.WHITE:
                if piece_type == chess.PAWN:
                    evaluation += PAWN_POSITION_TABLE[rank][file]
                elif piece_type == chess.KNIGHT:
                    evaluation += KNIGHT_POSITION_TABLE[rank][file]
                elif piece_type == chess.BISHOP:
                    evaluation += BISHOP_POSITION_TABLE[rank][file]
                elif piece_type == chess.ROOK:
                    evaluation += ROOK_POSITION_TABLE[rank][file]
                elif piece_type == chess.KING:
                    evaluation += KING_POSITION_TABLE[rank][file]
            else:
                if piece_type == chess.PAWN:
                    evaluation -= PAWN_POSITION_TABLE[7 - rank][file]  # Reverse rank for black
                elif piece_type == chess.KNIGHT:
                    evaluation -= KNIGHT_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.BISHOP:
                    evaluation -= BISHOP_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.ROOK:
                    evaluation -= ROOK_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.KING:
                    evaluation -= KING_POSITION_TABLE[7 - rank][file]

    
    try:
        
        prediction = predict_new_data_prob(board, MODEL_NUM)

        NB_score = prediction.get("1",0) - prediction.get("-1",0)

        print(f"NB score: {NB_score}")
        evaluation = EVALUATION_WEIGHT *evaluation + NB_WEIGHT*NB_score

        return evaluation
    
    except Exception as e:
        print(f"Naive Bayes prediction failed: {e}")
        print(board)
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
