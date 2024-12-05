import chess

# Create a new chess board
board = chess.Board()

board.set_fen("r1bqkbnr/pppppppp/2n2n2/3P4/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 10")


def evaluate_board(board):
    # Evaluate the board
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # The king's value is not considered in material balance
    }
    
    evaluation = 0
    for piece_type in piece_values:
        evaluation += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        evaluation -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    
    return evaluation

def alphaBetaMax(board, alpha, beta, depth):
    if depth == 0:
        return (evaluate_board(board),None)
    best_value = -float('inf')
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        score = alphaBetaMin(board, alpha, beta, depth - 1)[0]
        board.pop()
        if score > best_value:
            best_value = score
            best_move = move
            if score > alpha:
                alpha = score
        if score >= beta:
            break
    return (best_value, best_move)


def alphaBetaMin(board, alpha, beta, depth):
    if depth == 0:
        return (-evaluate_board(board),None)
    best_value = float('inf')
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        score = alphaBetaMax(board, alpha, beta, depth - 1)[0]
        board.pop()
        if score < best_value:
            best_value = score
            best_move = move
            if score < beta:
                beta = score
        if score <= alpha:
            break
    return (best_value, best_move)

# Find the best move for white
best_value, best_move = alphaBetaMax(board, -float('inf'), float('inf'), 6)
print("Best move for white:", best_move, "with value:", best_value)
    
