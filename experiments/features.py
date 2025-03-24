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

def calculate_material_balance(board):
    material_white = sum(PIECE_VALUES[piece] * len(board.pieces(piece, chess.WHITE)) for piece in PIECE_VALUES)
    material_black = sum(PIECE_VALUES[piece] * len(board.pieces(piece, chess.BLACK)) for piece in PIECE_VALUES)
    return material_white - material_black

def piece_mobility(board):
    player = board.turn
    player_moves = board.legal_moves.count()
    board_copy = board.copy(stack=False)
    board_copy.turn = not player
    opponent_moves = board_copy.legal_moves.count()
    mobility = player_moves - opponent_moves
    if player == chess.BLACK:
        mobility = -mobility
    return mobility

def control_of_center_small(board):
    center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
    white_pieces = sum(10 for square in center_squares 
                       if board.piece_at(square) and board.piece_at(square).color == chess.WHITE)
    black_pieces = sum(10 for square in center_squares 
                       if board.piece_at(square) and board.piece_at(square).color == chess.BLACK)
    return white_pieces - black_pieces

def control_of_center_large(board):
    center_squares = [
        chess.E4, chess.D4, chess.E5, chess.D5,  # Inner center
        chess.C3, chess.D3, chess.E3, chess.F3,  # Rank 3
        chess.C4, chess.F4, chess.C5, chess.F5,  # Files c, f
        chess.C6, chess.D6, chess.E6, chess.F6   # Rank 6
    ]
    white_pieces = sum(10 for square in center_squares 
                       if board.piece_at(square) and board.piece_at(square).color == chess.WHITE)
    black_pieces = sum(10 for square in center_squares 
                       if board.piece_at(square) and board.piece_at(square).color == chess.BLACK)
    return white_pieces - black_pieces

def king_attack_balance(board):
    w_king_sq = board.king(chess.WHITE)
    b_king_sq = board.king(chess.BLACK)
    w_attackers = board.attackers(chess.BLACK, w_king_sq) if w_king_sq is not None else set()
    b_attackers = board.attackers(chess.WHITE, b_king_sq) if b_king_sq is not None else set()
    return len(w_attackers) - len(b_attackers)

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
                    positional_score -= PAWN_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.KNIGHT:
                    positional_score -= KNIGHT_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.BISHOP:
                    positional_score -= BISHOP_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.ROOK:
                    positional_score -= ROOK_POSITION_TABLE[7 - rank][file]
                elif piece_type == chess.KING:
                    positional_score -= KING_POSITION_TABLE[7 - rank][file]
    return positional_score

def pawn_structure(board, color):
    pawn_squares = board.pieces(chess.PAWN, color)
    file_counts = [0] * 8
    for sq in pawn_squares:
        file_counts[chess.square_file(sq)] += 1
    doubled = sum(count - 1 for count in file_counts if count > 1)
    
    # Count isolated pawns: if a file has pawns and neither adjacent file has any.
    isolated = 0
    for file in range(8):
        if file_counts[file] > 0:
            left = file_counts[file - 1] if file - 1 >= 0 else 0
            right = file_counts[file + 1] if file + 1 < 8 else 0
            if left == 0 and right == 0:
                isolated += file_counts[file]
    return isolated, doubled

def castling_rights(board):
    return {
        'white_kingside_castle': board.has_kingside_castling_rights(chess.WHITE),
        'white_queenside_castle': board.has_queenside_castling_rights(chess.WHITE),
        'black_kingside_castle': board.has_kingside_castling_rights(chess.BLACK),
        'black_queenside_castle': board.has_queenside_castling_rights(chess.BLACK)
    }

def king_safety(board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return 0  # Undefined if the king is missing
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    adjacent_squares = [
        chess.square(f, r)
        for f in range(max(0, king_file - 1), min(8, king_file + 2))
        for r in range(max(0, king_rank - 1), min(8, king_rank + 2))
        if chess.square(f, r) != king_sq
    ]
    pawn_shield = sum(1 for sq in adjacent_squares 
                      if board.piece_at(sq) and 
                         board.piece_at(sq).piece_type == chess.PAWN and 
                         board.piece_at(sq).color == color)
    enemy = not color
    enemy_threats = sum(1 for sq in adjacent_squares if board.is_attacked_by(enemy, sq))
    return pawn_shield - enemy_threats

def add_king_safety_features(board, features):
    features['white_king_safety'] = king_safety(board, chess.WHITE)
    features['black_king_safety'] = king_safety(board, chess.BLACK)
    return features

def is_passed_pawn(board, pawn_sq, color):
    file = chess.square_file(pawn_sq)
    rank = chess.square_rank(pawn_sq)
    enemy_color = not color
    files_to_check = {file}
    if file - 1 >= 0:
        files_to_check.add(file - 1)
    if file + 1 < 8:
        files_to_check.add(file + 1)
    for enemy_sq in board.pieces(chess.PAWN, enemy_color):
        enemy_file = chess.square_file(enemy_sq)
        enemy_rank = chess.square_rank(enemy_sq)
        if enemy_file in files_to_check:
            if (color == chess.WHITE and enemy_rank > rank) or (color == chess.BLACK and enemy_rank < rank):
                return False
    return True

def count_passed_pawns(board, color):
    return sum(1 for pawn_sq in board.pieces(chess.PAWN, color) if is_passed_pawn(board, pawn_sq, color))

def add_passed_pawn_features(board, features):
    features['white_passed_pawns'] = count_passed_pawns(board, chess.WHITE)
    features['black_passed_pawns'] = count_passed_pawns(board, chess.BLACK)
    return features

def game_phase(board):
    # Define phase values for each piece type.
    phase_values = {
        chess.QUEEN: 4,
        chess.ROOK: 2,
        chess.BISHOP: 1,
        chess.KNIGHT: 1,
        chess.PAWN: 0,   # Pawns are less significant for phase determination
        chess.KING: 0    # Kings are not counted in phase value
    }
    
    # Calculate the total phase value for the starting position.
    # For one side: 1 Queen, 2 Rooks, 2 Bishops, 2 Knights.
    total_phase_one_side = (phase_values[chess.QUEEN] +
                              2 * phase_values[chess.ROOK] +
                              2 * phase_values[chess.BISHOP] +
                              2 * phase_values[chess.KNIGHT])
    total_phase = 2 * total_phase_one_side  # both sides
    
    # Compute the current phase value based on the pieces still on the board.
    current_phase = 0
    for piece_type, weight in phase_values.items():
        current_phase += weight * (
            len(board.pieces(piece_type, chess.WHITE)) +
            len(board.pieces(piece_type, chess.BLACK))
        )
    
    # Normalize the phase value: 1 indicates full material (opening), 0 indicates endgame.
    normalized_phase = current_phase / total_phase
    
    # Determine the game phase based on thresholds.
    if normalized_phase > 0.66:
        return 0  # Opening
    elif normalized_phase > 0.33:
        return 1  # Middlegame
    else:
        return 2  # Endgame


def add_game_phase_feature(board, features):
    features['game_phase'] = game_phase(board)
    return features


# Simple cache for board features keyed by FEN string.
_features_cache = {}

# def board_features(board, model=4):

#     material_balance = calculate_material_balance(board)
#     position_value = piece_position_value(board)
#     mobility = piece_mobility(board)
#     center_small = control_of_center_small(board)
#     center_large = control_of_center_large(board)
#     king_attack = king_attack_balance(board)
#     castling = castling_rights(board)
    
#     white_isolated, white_doubled = pawn_structure(board, chess.WHITE)
#     black_isolated, black_doubled = pawn_structure(board, chess.BLACK)

#     features = {
#         'material_balance': material_balance,
#         'position_value': position_value,
#         'piece_mobility': mobility,
#         'control_of_center_small': center_small,
#         'control_of_center_large': center_large,
#         'king_attack_balance': king_attack,
#         'white_isolated_pawns': white_isolated,
#         'black_isolated_pawns': black_isolated,
#         'white_doubled_pawns': white_doubled,
#         'black_doubled_pawns': black_doubled,
#     }
#     features.update(castling)
#     features = add_king_safety_features(board, features)
#     features = add_passed_pawn_features(board, features)
#     features = add_game_phase_feature(board, features)

#     return features

def board_features(board, model=4):
    if model == 0:
        return board_features_0(board)
    elif model == 1:
        return board_features_1(board)
    elif model == 2:
        return board_features_2(board)
    elif model == 3:
        return board_features_3(board)
    else:
        raise ValueError(f"Invalid model number: {model}")

def board_features_0(board):
    material_balance = calculate_material_balance(board)
    position_value = piece_position_value(board)
    mobility = piece_mobility(board)
    king_attack = king_attack_balance(board)

    features = {
        'material_balance': material_balance,
        'position_value': position_value,
        'piece_mobility': mobility,
        'king_attack_balance': king_attack
    }

    return features

def board_features_1(board):
    features = board_features_0(board)
    center_small = control_of_center_small(board)
    center_large = control_of_center_large(board)
    features['control_of_center_small'] = center_small
    features['control_of_center_large'] = center_large
    return features

def board_features_2(board):
    features = board_features_1(board)
    white_isolated, white_doubled = pawn_structure(board, chess.WHITE)
    black_isolated, black_doubled = pawn_structure(board, chess.BLACK)
    features['white_isolated_pawns'] = white_isolated
    features['black_isolated_pawns'] = black_isolated
    features['white_doubled_pawns'] = white_doubled
    features['black_doubled_pawns'] = black_doubled
    return features


def board_features_3(board):
    features = board_features_2(board)
    castling = castling_rights(board)
    features.update(castling)
    features = add_king_safety_features(board, features)
    features = add_game_phase_feature(board, features)
    return features
    
