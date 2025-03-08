import pygame as p
import chess
import chess.engine
import time
import random
from minimax import alphaBeta, evaluate
from chess_visual import VisualBoard


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
    


    while not board.is_game_over():
        
        if board.turn == chess.WHITE:
            if player_one == 0:
                move = random.choice(list(board.legal_moves))
            elif player_one == 1:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                move = result.move
            else:
                move = alphaBeta(board, -float("inf"), float("inf"), depth, True)[1]
        
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
            #print(board)
    

    engine.quit()
    print(board.result())
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





    while running:
        running = vboard.continue_game()

        if not board.is_game_over():
            if board.turn == chess.WHITE:
                if player_one == 0:
                    move = random.choice(list(board.legal_moves))
                elif player_one == 1:
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    move = result.move
                else:
                    move = alphaBeta(board, -float("inf"), float("inf"), depth, True)[1]
            
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
        
        elif x:
            x = False
            print(board.result())

    
    engine.quit()





def play_game_repeat(num, player_one, player_two, skill_level=1, depth=3):
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
    for i in range(num):
        result = play_game_blind(player_one, player_two, skill_level, depth)
        results[result] += 1

    print(results)



play_game_repeat(20, 1, 2, skill_level=1 ,depth=5)
