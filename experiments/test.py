import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load your results CSV file
df = pd.read_csv("chess_results.csv")

# Assuming 'moves_list' is the column name - replace with actual column name
moves_column = 'moves'  # Replace with your actual column name

# Convert string representations to actual lists and calculate lengths
if moves_column in df.columns:
    # Convert string representations to actual lists
    moves_lists = df[moves_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Calculate the number of moves in each list
    move_counts = moves_lists.apply(len)
    
    # Find the maximum
    max_moves = move_counts.max()
    
    print(f"Maximum number of moves in a game: {max_moves}")
    print(f"Game with maximum moves: {df.iloc[move_counts.idxmax()]}")
else:
    print(f"Column '{moves_column}' not found in the dataframe")
    print("Available columns:", df.columns.tolist())