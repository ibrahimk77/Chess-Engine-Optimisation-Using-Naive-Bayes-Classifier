import pygame as p
import chess
import chess.engine
import time
import random
from minimax import alphaBeta
#from minimax_NB_eval import alphaBeta
#from minimax_NB_integrated import alphaBeta
from chess_visual import VisualBoard


random.seed(1)
DEPTH = 4
LEVEL = 0
PLAYER_ONE = 2
PLAYER_TWO = 1

# :param player_one: The first player.
#     0 -> random, 1-> stockfish, 2 -> minimax


STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Path to the Stockfish engine


def play_game_blind(player_one, player_two, skill_level=1, depth=3):
    """
    Play a game of chess between two players.

    :param skill_level: The skill level of the Stockfish engine.
    :param player_one: The first player.
        0 -> random, 1-> stockfish, 2 -> minimax
    :param player_two: The second player

    """

    players = {0: "Random", 1: "Stockfish", 2: "Minimax"}

    print(f"{players[player_one]} vs {players[player_two]}")

    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Skill Level": skill_level})
    
    total_time = 0
    moves = 0


    while not board.is_game_over():
        
        if board.turn == chess.WHITE:
            start = time.time()

            if player_one == 0:
                move = random.choice(list(board.legal_moves))
            elif player_one == 1:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                move = result.move
            else:
                move = alphaBeta(board, -float("inf"), float("inf"), depth, True)[1]
            end = time.time()
            total_time += end - start
        
        else:
            if player_two == 0:
                move = random.choice(list(board.legal_moves))
            elif player_two == 1:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                move = result.move
            else:
                move = alphaBeta(board, -float("inf"), float("inf"), depth, False)[1]

        if move:
            board.push(move)
            moves += 1
            #print(board)
    

    engine.quit()
    print(board.result())
    print(f"Average time taken for each move: {total_time/moves}")
    print(count_pieces(board))
    return board.result()




def play_game_visual(player_one, player_two, skill_level=1, depth=3):

    board = chess.Board()
    vboard = VisualBoard()
    vboard.initialise_board()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Skill Level": skill_level})
    running = True
    x = True

    players = {0: "Random", 1: "Stockfish", 2: "Minimax"}

    print(f"{players[player_one]} vs {players[player_two]}")


    total_time = 0
    moves = 0

    white_score = 0
    black_score = 0


    while running:
        running = vboard.continue_game()

        if not board.is_game_over():
            if board.turn == chess.WHITE:
                start = time.time()
                if player_one == 0:
                    move = random.choice(list(board.legal_moves))
                elif player_one == 1:
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    move = result.move
                else:
                    move = alphaBeta(board, -float("inf"), float("inf"), depth, True)[1]
                end = time.time()
                total_time += end - start
            
            else:
                if player_two == 0:
                    move = random.choice(list(board.legal_moves))
                elif player_two == 1:
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    move = result.move
                else:
                    move = alphaBeta(board, -float("inf"), float("inf"), depth, False)[1]

            if move:
                board.push(move)
                vboard.drawGame(board)
                analysis = engine.analyse(board, chess.engine.Limit(time=0.1))
                score = analysis['score']
                if not score.is_mate():
                    if moves % 2 == 0:
                        white_score += score.relative.score()
                    else:
                        black_score -= score.relative.score()
                moves += 1
                    
        
        elif x:
            x = False
            print(board.result())
            print(f"Average time taken for each move: {total_time/moves}")
            print(f"White score: {white_score/(moves/2)}")
            print("Total moves: ", moves)
    engine.quit()





def play_game_repeat(player_one, player_two, skill_level=1, depth=3, num=10):
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for i in range(num):
        result = play_game_blind(player_one, player_two, skill_level, depth)
        results[result] += 1

    print(results)


def count_pieces(board):
    piece_counts = {
        'white': 0,
        'black': 0
    }
    for piece_type in chess.PIECE_TYPES:
        piece_counts['white'] += len(board.pieces(piece_type, chess.WHITE))
        piece_counts['black'] += len(board.pieces(piece_type, chess.BLACK))
    return piece_counts


#play_game_repeat(PLAYER_ONE, PLAYER_TWO, skill_level=LEVEL ,depth=DEPTH, num=10)

play_game_visual(PLAYER_ONE, PLAYER_TWO, skill_level=LEVEL, depth=DEPTH)
