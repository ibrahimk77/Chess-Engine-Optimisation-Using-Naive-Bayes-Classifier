import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0 
}

def calculate_material_balance(board):
    material_white = sum(PIECE_VALUES[piece] * len(board.pieces(piece, chess.WHITE)) for piece in PIECE_VALUES)
    material_black = sum(PIECE_VALUES[piece] * len(board.pieces(piece, chess.BLACK)) for piece in PIECE_VALUES)
    return material_white - material_black

def piece_mobility(board):
    return sum(1 for _ in board.legal_moves)

def control_of_center_small(board):
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    
    white_pieces = sum(10 for square in center_squares 
                      if board.piece_at(square) 
                      and board.piece_at(square).color == chess.WHITE)
    
    black_pieces = sum(10 for square in center_squares 
                      if board.piece_at(square) 
                      and board.piece_at(square).color == chess.BLACK)
    
    return white_pieces - black_pieces

def control_of_center_large(board):
    center_squares = [
        chess.E4, chess.D4, chess.E5, chess.D5,  # Inner center
        chess.C3, chess.D3, chess.E3, chess.F3,  # Rank 3
        chess.C4, chess.F4, chess.C5, chess.F5,  # Files c,f
        chess.C6, chess.D6, chess.E6, chess.F6   # Rank 6
    ]
    
    white_pieces = sum(10 for square in center_squares 
                      if board.piece_at(square) 
                      and board.piece_at(square).color == chess.WHITE)
    
    black_pieces = sum(10 for square in center_squares 
                      if board.piece_at(square) 
                      and board.piece_at(square).color == chess.BLACK)
    
    return white_pieces - black_pieces


def king_attack_balance(board):
    #positive = white under attack
    #negative = black under attack
    w_king_square = board.king(chess.WHITE)
    b_king_square = board.king(chess.BLACK)

    w_attackers = board.attackers(chess.BLACK, w_king_square)
    b_attackers = board.attackers(chess.WHITE, b_king_square)

    return len(w_attackers) - len(b_attackers)
    
def king_position(board):
    pass
