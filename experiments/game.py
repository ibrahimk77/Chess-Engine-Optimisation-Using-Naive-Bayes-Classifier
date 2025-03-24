import pygame as p
import chess
import chess.engine
import time
import random


from minimax import alphaBeta
from minimax_NB_sub import alphaBeta_sub
from minimax_NB_integrated import alphaBeta_integrated
from minimax_NB_ordering import alphaBeta_ordering

DEPTH = 3
LEVEL = 0
PLAYER_ONE = 2
PLAYER_TWO = 1
STOCKFISH_TIME = 0.1
ANALYSIS_TIME = 0.1
BLUNDER_THRESHOLD = 300
GOOD_MOVE_THRESHOLD = 200
DEFAULT_NB_WEIGHT = 0

STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Path to the Stockfish engine


def play(model, scaler, opponent, game_num, implementation, feature, nb_weight=DEFAULT_NB_WEIGHT):

    start_time = time.time()
    moves_times = []
    confidences = []
    nodes_explored = []
    mobilities = []
    piece_balances = []
    blunders = []
    good_moves = []
    stockfish_evals = []
    moves = []



    stats = []

    random.seed(game_num)
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Skill Level": LEVEL})
    stockfish_current_eval = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME))["score"].white().score(mate_score=10000)


    while not board.is_game_over():        

        
        if board.turn == chess.WHITE:
            move_start_time = time.time()

            confidence, move, nodes = get_alphaBeta_move(board, model, scaler, implementation, True, feature, nb_weight)

            move_end_time = time.time()

            confidences.append(confidence)
            moves_times.append(move_end_time - move_start_time)
            nodes_explored.append(nodes)
            mobilities.append(len(list(board.legal_moves)))
            piece_balances.append(sum([len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)) for piece in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]]))

            board.push(move)
            moves.append(move.uci())

            stockfish_eval = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))["score"].white().score(mate_score=10000)
            eval_diff = stockfish_eval - stockfish_current_eval
            if eval_diff < -BLUNDER_THRESHOLD:
                blunders.append(eval_diff)
            else:
                blunders.append(0)
            if eval_diff > GOOD_MOVE_THRESHOLD:
                good_moves.append(eval_diff)
            else:
                good_moves.append(0)
            stockfish_current_eval = stockfish_eval
            stockfish_evals.append(stockfish_eval)


        else:
            if opponent == 'random':
                move = random.choice(list(board.legal_moves))
            elif opponent == 'stockfish':
                result = engine.play(board, chess.engine.Limit(time=STOCKFISH_TIME))
                move = result.move
            else:
                _, move, _ = get_alphaBeta_move(board, model, scaler, implementation, False, feature, nb_weight)
            board.push(move)
            stockfish_current_eval = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))["score"].white().score(mate_score=10000)
                

    engine.quit()
    print(board.result())

    confidences = [float(c) for c in confidences]
    nodes_explored = [int(n) for n in nodes_explored]
    mobilities = [int(m) for m in mobilities]
    blunders = [int(b) for b in blunders]
    good_moves = [int(g) for g in good_moves]
    stockfish_evals = [int(s) for s in stockfish_evals]


        
    whole_stats ={
        "moves_times": moves_times,
        "avg_move_times": sum(moves_times) / len(moves_times),
        "confidences": confidences,
        "nodes_explored": nodes_explored,
        "avg_nodes_explored": sum(nodes_explored) / len(nodes_explored),
        "mobilities": mobilities,
        "avg_mobilities": sum(mobilities) / len(mobilities),
        "piece_balances": piece_balances,
        "avg_piece_balances": sum(piece_balances) / len(piece_balances),
        "blunders": blunders,
        "good_moves": good_moves,
        "stockfish_evals": stockfish_evals,
        "avg_stockfish_evals": sum(stockfish_evals) / len(stockfish_evals),
        "nb_weight": nb_weight,
        "total_time": time.time() - start_time,
        "moves": moves,
        "depth": DEPTH,
        "result": board.result(),
        "phase": "whole"
    }
     
    total_white_moves = len(moves)
    phase_stats = []
    if total_white_moves > 0:
        opening_end = int(total_white_moves * 0.25)
        mid_end = int(total_white_moves * 0.75)

        def compute_phase_stats(phase_name, indices):
            if not indices:
                return None
            return {
                "moves_times": [moves_times[i] for i in indices],
                "avg_move_times": sum(moves_times[i] for i in indices) / len(indices),
                "confidences": [confidences[i] for i in indices],
                "nodes_explored": [nodes_explored[i] for i in indices],
                "avg_nodes_explored": sum(nodes_explored[i] for i in indices) / len(indices) if indices else 0,
                "mobilities": [mobilities[i] for i in indices],
                "avg_mobilities": sum(mobilities[i] for i in indices) / len(indices) if indices else 0,
                "piece_balances": [piece_balances[i] for i in indices],
                "avg_piece_balances": sum(piece_balances[i] for i in indices) / len(indices) if indices else 0,
                "blunders": [blunders[i] for i in indices],
                "good_moves": [good_moves[i] for i in indices],
                "stockfish_evals": [stockfish_evals[i] for i in indices],
                "avg_stockfish_evals": sum(stockfish_evals[i] for i in indices) / len(indices) if indices else 0,
                "nb_weight": nb_weight,
                "moves": [moves[i] for i in indices],
                "depth": DEPTH,
                "result": board.result(),
                "phase": phase_name,
                "total_time": None  # You may choose not to record per-phase time here.
            }

        opening_indices = list(range(0, opening_end))
        mid_indices = list(range(opening_end, mid_end))
        end_indices = list(range(mid_end, total_white_moves))

        opening_stats = compute_phase_stats("opening", opening_indices)
        mid_stats = compute_phase_stats("midgame", mid_indices)
        end_stats = compute_phase_stats("endgame", end_indices)

        for phase_dict in (opening_stats, mid_stats, end_stats):
            if phase_dict is not None:
                phase_stats.append(phase_dict)

    # Combine phase stats and whole-game stats into one list.
    all_stats = phase_stats + [whole_stats]
    return all_stats




def get_alphaBeta_move(board, model, scaler, implementation, is_white, feature, nb_weight):

    if implementation == 'substitution':
        confidence, move, nodes_explored = alphaBeta_sub(board, -float("inf"), float("inf"), DEPTH, True, model, scaler, feature)
    elif implementation == 'integration':
        confidence, move, nodes_explored = alphaBeta_integrated(board, -float("inf"), float("inf"), DEPTH, True, model, scaler, feature, nb_weight)
    elif implementation == 'ordering':
        confidence, move, nodes_explored = alphaBeta_ordering(board, -float("inf"), float("inf"), DEPTH, True, model, scaler, feature)
    else:
        confidence, move, nodes_explored = alphaBeta(board, -float("inf"), float("inf"), DEPTH, True)

    return confidence, move, nodes_explored
