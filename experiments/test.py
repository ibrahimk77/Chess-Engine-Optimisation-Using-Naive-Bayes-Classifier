import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load your results CSV file
df = pd.read_csv("chess_results.csv")

# Define the game phases you want to plot (e.g., opening, midgame, endgame)
phases = ["opening", "midgame", "endgame"]

plt.figure(figsize=(12, 8))

# Loop over each phase
for phase in phases:
    # Filter the dataframe for the current phase
    phase_df = df[df["phase"] == phase]
    if not phase_df.empty:
        # If there are multiple rows per phase, you can choose to plot one game
        # or average them over move index. Here, we'll simply take the first row.
        row = phase_df.iloc[0]
        # Convert the confidences string back into a Python list.
        confidences = ast.literal_eval(row["confidences"])
        # Create an x-axis (move number within the phase)
        moves = list(range(1, len(confidences) + 1))
        plt.plot(moves, confidences, marker="o", label=phase)

plt.xlabel("Move Number (in Phase)")
plt.ylabel("Confidence")
plt.title("Confidence Over Time in Different Game Phases")
plt.legend(title="Phase")
plt.grid(True)
plt.tight_layout()
plt.show()
