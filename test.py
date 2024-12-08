import pygame as p
import chess
import chess.engine
from chess_engine import ChessGame
import time

# Constants for the board and pieces
WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}
STOCKFISH_PATH = "/stockfish"  # Path to the Stockfish engine

# Load images for the pieces
def loadImages():
    pieces = ['wp', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bp', 'bN', 'bB', 'bR', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))

# Draw the game state
def drawGame(screen, game):
    drawBoard(screen)
    drawPieces(screen, game)

# Draw the chessboard
def drawBoard(screen):
    colors = [p.Color("white"), p.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            p.draw.rect(screen, color, p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

# Draw the pieces on the board
def drawPieces(screen, game):
    board_array = game.fen_to_2d_array()
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board_array[r][c]
            if piece != '--':  # Draw only non-empty squares
                screen.blit(IMAGES[piece], p.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

# Main function to run the Pygame window
def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    game = ChessGame()  # Initialize your custom ChessGame class
    loadImages()
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    running = True

    while running:
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False

        # AI makes a move and updates the board
        if not game.board.is_game_over():
            if game.board.turn == chess.WHITE:
                best_value, best_move = alphaBetaMax(game.board, -float("inf"), float("inf"), 5)
            else:
                result = engine.play(game.board, chess.engine.Limit(time=1))
                best_move = result.move
            # best_value, best_move = (
            #     alphaBetaMax(game.board, -float("inf"), float("inf"), 5)
            #     if game.board.turn == chess.WHITE
            #     else alphaBetaMin(game.board, -float("inf"), float("inf"), 5)
            # )
            if best_move:
                game.board.push(best_move)  # Update the board with the best move

                drawGame(screen, game)  # Redraw the board after each move
                p.display.flip()
                #time.sleep(1)


        drawGame(screen, game)  # Redraw the board after each move
        clock.tick(MAX_FPS)
        p.display.flip()

# Alpha-beta pruning functions
def alphaBetaMax(board, alpha, beta, depth):
    if depth == 0 or board.is_game_over():
        return (evaluate_board(board), None)
    best_value = -float("inf")
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        score = alphaBetaMin(board, alpha, beta, depth - 1)[0]
        board.pop()
        if score > best_value:
            best_value = score
            best_move = move
            alpha = max(alpha, score)
        if score >= beta:
            break
    return (best_value, best_move)

def alphaBetaMin(board, alpha, beta, depth):
    if depth == 0 or board.is_game_over():
        return (-evaluate_board(board), None)
    best_value = float("inf")
    best_move = None

    for move in board.legal_moves:
        board.push(move)
        score = alphaBetaMax(board, alpha, beta, depth - 1)[0]
        board.pop()
        if score < best_value:
            best_value = score
            best_move = move
            beta = min(beta, score)
        if score <= alpha:
            break
    return (best_value, best_move)

def evaluate_board(board):
    # Evaluate the board
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,  # The king's value is not considered in material balance
    }

    evaluation = 0
    for piece_type in piece_values:
        evaluation += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        evaluation -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    return evaluation

# ChessGame class that holds the board and converts FEN to a 2D array
class ChessGame:
    def __init__(self):
        self.board = chess.Board()
        self.board.set_fen(
            "r1bqkbnr/pppppppp/2n2n2/3P4/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 10"
        )

    def fen_to_2d_array(self):
        # Convert the FEN string into a 2D array for the board
        fen = self.board.fen()
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

# Run the main program
main()
