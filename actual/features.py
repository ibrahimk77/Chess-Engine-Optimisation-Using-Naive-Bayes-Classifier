import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0 
}

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

#TODO: King position in endgame
#TODO: Consider other features









def calculate_material_balance(board):
    material_white = sum(PIECE_VALUES[piece] * len(board.pieces(piece, chess.WHITE)) for piece in PIECE_VALUES)
    material_black = sum(PIECE_VALUES[piece] * len(board.pieces(piece, chess.BLACK)) for piece in PIECE_VALUES)
    return material_white - material_black

def piece_mobility(board):
    player = board.turn

    count = sum(1 for _ in board.legal_moves)
    board.turn = not player
    count -= sum(1 for _ in board.legal_moves)

    board.turn = player

    if player == chess.BLACK:
        count = -count
    
    return count

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
    #TODO: Finish
    pass


def piece_position_value(board):

    positional_score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            
            file = chess.square_file(square)
            rank = chess.square_rank(square)

            if piece.color == chess.WHITE:
                if piece_type == chess.PAWN:
                    positional_score += PAWN_POSITION_TABLE[rank][file]
                elif piece_type == chess.KNIGHT:
                    positional_score += KNIGHT_POSITION_TABLE[rank][file]
                elif piece_type == chess.BISHOP:
                    positional_score += BISHOP_POSITION_TABLE[rank][file]
                elif piece_type == chess.ROOK:
                    positional_score += ROOK_POSITION_TABLE[rank][file]
                elif piece_type == chess.KING:
                    positional_score += KING_POSITION_TABLE[rank][file]
            else:
                if piece_type == chess.PAWN:
                    positional_score -= PAWN_POSITION_TABLE[7 - rank][file]  # Reverse rank for black
                elif piece_type == chess.KNIGHT:
                    positional_score -= KNIGHT_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.BISHOP:
                    positional_score -= BISHOP_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.ROOK:
                    positional_score -= ROOK_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.KING:
                    positional_score -= KING_POSITION_TABLE[7 - rank][file]
    return positional_score


def isolated_pawns(board, color):
    isolated = 0
    pawn_sqaures = board.pieces(chess.PAWN, color)
    pawn_files = [chess.square_file(square) for square in pawn_sqaures]

    for sq in pawn_sqaures:
        file = chess.square_file(sq)
        adjacent_files = []

        if file - 1 >= 0:
            adjacent_files.append(file - 1)
        if file + 1 <= 7:
            adjacent_files.append(file + 1)
        
        if not any(file in pawn_files for file in adjacent_files):
            isolated += 1
    
    return isolated

def doubled_pawns(board, color):
    doubled = 0
    file_counts = {i: 0 for i in range(8)}

    for square in board.pieces(chess.PAWN, color):
        file_counts[chess.square_file(square)] += 1

    for count in file_counts.values():
        if count > 1:
            doubled += count - 1

    return doubled

def castling_rights(board):
    white_kingside = board.has_kingside_castling_rights(chess.WHITE)
    white_queenside = board.has_queenside_castling_rights(chess.WHITE)
    black_kingside = board.has_kingside_castling_rights(chess.BLACK)
    black_queenside = board.has_queenside_castling_rights(chess.BLACK)

    return {
        'white_kingside_castle': white_kingside,
        'white_queenside_castle': white_queenside,
        'black_kingside_castle': black_kingside,
        'black_queenside_castle': black_queenside
    }




def board_features(board):
    material_balance = calculate_material_balance(board)
    position_value = piece_position_value(board)
    mobility = piece_mobility(board)
    center_small = control_of_center_small(board)
    center_large = control_of_center_large(board)
    king_attack = king_attack_balance(board)
    move_number = board.fullmove_number
    w_isolated = isolated_pawns(board, chess.WHITE)
    b_isolated = isolated_pawns(board, chess.BLACK)
    w_doubled = doubled_pawns(board, chess.WHITE)
    b_doubled = doubled_pawns(board, chess.BLACK)
    castling = castling_rights(board)

    features = {
        'material_balance': material_balance,
        'position_value': position_value,
        'piece_mobility': mobility,
        'control_of_center_small': center_small,
        'control_of_center_large': center_large,
        'king_attack_balance': king_attack,
        'white_isolated_pawns': w_isolated,
        'black_isolated_pawns': b_isolated,
        'white_doubled_pawns': w_doubled,
        'black_doubled_pawns': b_doubled,
        #'move_number': move_number
    }

    features.update(castling)

    return features


