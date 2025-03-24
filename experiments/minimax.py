import chess

def alphaBeta(board, alpha, beta, depth, is_maximising):
    nodes_explored = 1  
    if depth == 0 or board.is_game_over():
        return evaluate(board), None, nodes_explored

    legal = list(board.legal_moves)

    best_move = legal[0]
    if is_maximising:
        value = -float("inf")
        for move in legal:
            board.push(move)
            eval_val, _, nodes_child = alphaBeta(board, alpha, beta, depth - 1, False)
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
        for move in board.legal_moves:
            board.push(move)
            eval_val, _, nodes_child = alphaBeta(board, alpha, beta, depth - 1, True)
            nodes_explored += nodes_child 
            board.pop()
            if eval_val < value:
                value = eval_val
                best_move = move
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move, nodes_explored


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
        [5, 10, 10, -20, -20, 10, 10, 5],
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

    return evaluation
