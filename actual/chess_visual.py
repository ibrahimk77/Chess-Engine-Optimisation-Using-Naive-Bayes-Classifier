import pygame as p
import chess
import chess.engine
import time
import random

# Constants for the board and pieces
WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}
STOCKFISH_PATH = "stockfish/stockfish-windows-x86-64-avx2.exe"  # Path to the Stockfish engine


class VisualBoard:
    def initialise_board(self):
        p.init()
        self.screen = p.display.set_mode((WIDTH, HEIGHT))
        clock = p.time.Clock()
        self.screen.fill(p.Color("white"))
        self.loadImages()

    def continue_game(self):
        for e in p.event.get():
            if e.type == p.QUIT:
                return False
        return True


    # Load images for the pieces
    def loadImages(self):
        pieces = ['wp', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bp', 'bN', 'bB', 'bR', 'bQ', 'bK']
        for piece in pieces:
            IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))

    # Draw the game state
    def drawGame(self, board):
        self.drawBoard()
        self.drawPieces(board)
        p.display.flip()

    # Draw the chessboard
    def drawBoard(self):
        colors = [p.Color("white"), p.Color("gray")]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[(r + c) % 2]
                p.draw.rect(self.screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

    # Draw the pieces on the board
    def drawPieces(self, board):
        board_array = self.fen_to_2d_array(board)
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                piece = board_array[r][c]
                if piece != '--':  # Draw only non-empty squares
                    self.screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))


    def fen_to_2d_array(self, board):
            # Convert the FEN string into a 2D array for the board
            fen = board.fen()
            board_fen, _ = fen.split(" ", 1)
            piece_mapping = {
                "p": "bp", "n": "bN", "b": "bB", "r": "bR", "q": "bQ", "k": "bK",
                "P": "wp", "N": "wN", "B": "wB", "R": "wR", "Q": "wQ", "K": "wK",
            }
            rows = board_fen.split("/")
            board = []
            for row in rows:
                parsed_row = []
                for char in row:
                    if char.isdigit():
                        parsed_row.extend(["--"] * int(char))
                    else:
                        parsed_row.append(piece_mapping[char])
                board.append(parsed_row)
            return board


