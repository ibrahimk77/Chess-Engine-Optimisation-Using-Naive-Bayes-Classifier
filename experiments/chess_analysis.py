"""
chess_analysis.py

Revised version to handle problematic CSV columns:
- Parses list columns with a try/except and regex replacement for tokens (e.g., 'inf')
- Creates additional mean columns (e.g., mean_blunders) for list-valued columns
- Implements comparisons and plots as previously suggested.
"""

import ast
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
# 1) HELPER FUNCTIONS
# -------------------------------------------------------

def parse_list_columns(df, list_columns):
    """
    Convert columns in `list_columns` from string representations of lists
    into actual Python lists. Replaces 'inf', '-inf', and similar tokens with
    appropriate float values.
    """
    for col in list_columns:
        if col in df.columns:
            def convert_func(val):
                # If not a string, return as is.
                if not isinstance(val, str):
                    return val
                s = val.strip()
                if not s.startswith('['):
                    return val
                # Replace 'inf' with a large number, and '-inf' with a very small number.
                s = re.sub(r'\binf\b', '1e999999', s)
                s = re.sub(r'\b-inf\b', '-1e999999', s)
                try:
                    return ast.literal_eval(s)
                except Exception as e:
                    # Option: log error or return None
                    return None
            df[col] = df[col].apply(convert_func)
    return df

def safe_mean(arr):
    """
    Compute the mean of a list or return np.nan if invalid.
    """
    if isinstance(arr, list) and len(arr) > 0:
        # Filter out non-numeric values if necessary
        try:
            numeric_vals = [float(x) for x in arr if isinstance(x, (int, float))]
            if numeric_vals:
                return np.mean(numeric_vals)
        except Exception:
            return np.nan
    return np.nan

def create_mean_columns(df, list_columns):
    """
    For each column in list_columns, create a new column 'mean_<col>' that stores the mean of the list.
    """
    for col in list_columns:
        new_col = 'mean_' + col
        df[new_col] = df[col].apply(lambda x: safe_mean(x) if isinstance(x, list) else np.nan)
    return df

# -------------------------------------------------------
# 2) MAIN LOADING AND DATA PREPARATION
# -------------------------------------------------------

def load_and_prepare_data(csv_path='chess_results.csv'):
    # Read CSV
    df = pd.read_csv(csv_path)

    # List columns that are expected to be list-like
    list_columns = [
        'moves_times', 'confidences', 'nodes_explored', 'mobilities',
        'piece_balances', 'blunders', 'good_moves', 'stockfish_evals', 'moves'
    ]
    df = parse_list_columns(df, list_columns)
    df = create_mean_columns(df, ['blunders', 'moves_times', 'confidences', 'nodes_explored', 'mobilities', 'piece_balances', 'good_moves', 'stockfish_evals'])
    
    # Ensure numeric columns are proper numeric type
    numeric_cols = ['avg_move_times', 'avg_nodes_explored', 'avg_mobilities', 'avg_piece_balances', 'avg_stockfish_evals', 'nb_weight', 'total_time']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# -------------------------------------------------------
# 3) PLOTTING FUNCTIONS FOR VARIOUS COMPARISONS
# -------------------------------------------------------

def plot_implementation_comparison(df):
    """
    Compare Substitution vs Integration across various metrics.
    Uses 'avg_move_times', 'avg_stockfish_evals', and the computed 'mean_blunders'.
    """
    sub_df = df[df['implementation'].isin(['substitution','integration'])].copy()
    metrics = ['avg_move_times', 'avg_stockfish_evals', 'mean_blunders']
    group = sub_df.groupby('implementation')[metrics].mean()
    
    group.plot(kind='bar', figsize=(8,5))
    plt.title('Implementation Comparison: Substitution vs Integration')
    plt.ylabel('Average Metric Value')
    plt.xticks(rotation=0)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_nb_weight_comparison(df):
    """
    For rows where implementation == 'integration', plot average nodes_explored vs. nb_weight.
    """
    integ_df = df[df['implementation'] == 'integration'].copy()
    if 'avg_nodes_explored' in integ_df.columns:
        comp = integ_df.groupby('nb_weight')['avg_nodes_explored'].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.lineplot(data=comp, x='nb_weight', y='avg_nodes_explored', marker='o')
        plt.title('NB Weight vs. Avg Nodes Explored (Integration)')
        plt.xlabel('Naive Bayes Weight')
        plt.ylabel('Avg Nodes Explored')
        plt.tight_layout()
        plt.show()



def plot_opponent_comparison(df):
    """
    Compares performance by opponent (random vs. stockfish) in two ways:
    
      1. Stacked bar chart for game results with mapped labels:
         "MMNB win", "MMNB loss", and "tie".
         
      2. Bar chart for average nonzero blunders per game (computed from the 'blunders'
         list while excluding 0 values).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # ----- Part 1: Game Results with Mapped Labels -----
    # Define mapping: adjust if necessary
    result_mapping = {"1-0": "MMNB win", "0-1": "MMNB loss", "1/2-1/2": "tie"}
    # Create new column 'result_mapped'
    df['result_mapped'] = df['result'].map(result_mapping).fillna(df['result'])
    
    # Group by opponent and mapped result, then plot a stacked bar chart.
    result_counts = df.groupby(['opponent', 'result_mapped']).size().unstack(fill_value=0)
    plt.figure(figsize=(8,5))
    result_counts.plot(kind='bar', stacked=True)
    plt.title('Game Results by Opponent')
    plt.ylabel('Number of Games')
    plt.xlabel('Opponent')
    plt.xticks(rotation=0)
    plt.legend(title='Result')
    plt.tight_layout()
    plt.show()
    
    # ----- Part 2: Average Nonzero Blunders per Game -----
    # Helper function to compute average of nonzero values in a list.
    def avg_nonzero(vals):
        if isinstance(vals, list) and len(vals) > 0:
            nonzero = [x for x in vals if x != 0]
            if nonzero:
                return np.mean(nonzero)
        return np.nan

    # Create a new column that computes the average nonzero blunders per game.
    df['avg_nonzero_blunders'] = df['blunders'].apply(avg_nonzero)
    
    # Group by opponent and compute the mean of these average nonzero blunders.
    blunder_mean = df.groupby('opponent')['avg_nonzero_blunders'].mean().reset_index()
    
    plt.figure(figsize=(6,4))
    sns.barplot(data=blunder_mean, x='opponent', y='avg_nonzero_blunders', palette='viridis')
    plt.title('Average Blunders per Game by Opponent')
    plt.xlabel('Opponent')
    plt.ylabel('Average Blunder values')
    plt.tight_layout()
    plt.show()

def plot_feature_selection_comparison(df):
    """
    Compare feature_selection values (e.g., 0 vs. 3) for average move times.
    """
    if 'feature_selection' in df.columns:
        df['feature_selection'] = df['feature_selection'].astype(str)
        group = df.groupby('feature_selection')['avg_move_times'].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x='feature_selection', y='avg_move_times')
        plt.title('Avg Move Times by Feature Selection')
        plt.xlabel('Feature Selection')
        plt.ylabel('Avg Move Times')
        plt.tight_layout()
        plt.show()

def plot_dataset_comparison(df):
    """
    Compare performance across datasets using avg_stockfish_evals.
    """
    if 'dataset' in df.columns and 'avg_stockfish_evals' in df.columns:
        group = df.groupby('dataset')['avg_stockfish_evals'].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x='dataset', y='avg_stockfish_evals')
        plt.title('Avg Stockfish Eval by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Avg Stockfish Eval')
        plt.tight_layout()
        plt.show()

def plot_phase_comparison(df):
    """
    Compare average mobilities across phases.
    """
    if 'phase' in df.columns and 'avg_mobilities' in df.columns:
        group = df.groupby('phase')['avg_mobilities'].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x='phase', y='avg_mobilities')
        plt.title('Avg Mobilities by Phase')
        plt.xlabel('Phase')
        plt.ylabel('Avg Mobilities')
        plt.tight_layout()
        plt.show()

def plot_colour_comparison(df):
    """
    Compare average blunders by colour.
    """
    if 'colour' in df.columns and 'mean_blunders' in df.columns:
        group = df.groupby('colour')['mean_blunders'].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x='colour', y='mean_blunders')
        plt.title('Avg Blunders by Colour')
        plt.xlabel('Colour')
        plt.ylabel('Avg Blunders')
        plt.tight_layout()
        plt.show()

def plot_correlation_matrix(df):
    """
    Create a correlation heatmap for selected numeric columns.
    """
    numeric_cols = ['avg_move_times', 'avg_nodes_explored', 'avg_stockfish_evals', 'mean_blunders', 'nb_weight', 'total_time']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    corr_df = df[numeric_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_time_move_evolution(df):
    """
    Plot an example game evolution for stockfish_evals.
    """
    example_row = df.iloc[0]
    if isinstance(example_row.get('stockfish_evals', None), list):
        stock_evals = example_row['stockfish_evals']
        plt.figure(figsize=(7,4))
        plt.plot(stock_evals, marker='o')
        plt.title('Stockfish Evaluations Over Moves (Example Game)')
        plt.xlabel('Move Index')
        plt.ylabel('Evaluation')
        plt.tight_layout()
        plt.show()

def plot_confidence_usage(df):
    """
    Plot average confidence by game result.
    """
    if 'confidences' in df.columns:
        df['mean_conf'] = df['confidences'].apply(lambda x: safe_mean(x) if isinstance(x, list) else np.nan)
    if 'result' in df.columns:
        group = df.groupby('result')['mean_conf'].mean().reset_index()
        plt.figure(figsize=(6,4))
        sns.barplot(data=group, x='result', y='mean_conf')
        plt.title('Average Confidence by Game Result')
        plt.xlabel('Result')
        plt.ylabel('Mean Confidence')
        plt.tight_layout()
        plt.show()

def plot_nodes_vs_performance(df):
    """
    Scatter plot of avg_nodes_explored vs. avg_stockfish_evals,
    colored by implementation and shaped by opponent.
    """
    if 'avg_nodes_explored' in df.columns and 'avg_stockfish_evals' in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(data=df, x='avg_nodes_explored', y='avg_stockfish_evals',
                        hue='implementation', style='opponent', s=80)
        plt.title('Nodes Explored vs. Stockfish Eval')
        plt.xlabel('Avg Nodes Explored')
        plt.ylabel('Avg Stockfish Eval')
        plt.tight_layout()
        plt.show()


def plot_implementation_comparison(df):
    """
    1) First chart: Compare Substitution vs. Integration 
       for 'avg_stockfish_evals' and 'avg_move_times'.

    2) Second chart: Compare total (or average) number of blunders 
       for Substitution vs. Integration, where a 'blunder' 
       is a non-zero entry in the 'blunders' list.
    """

    # Filter for just substitution/integration
    sub_df = df[df['implementation'].isin(['substitution','integration'])].copy()

    # ------------------------------
    # (A) CHART: avg_stockfish_evals and avg_move_times
    # ------------------------------
    metrics = ['avg_stockfish_evals', 'avg_move_times']
    group_metrics = sub_df.groupby('implementation')[metrics].mean()
    group_metrics = group_metrics.sort_index()  # ensure consistent order

    # Prepare x-axis
    x_labels = group_metrics.index.tolist()
    x = np.arange(len(x_labels))

    # Extract each metric
    evals_vals = group_metrics['avg_stockfish_evals']
    times_vals = group_metrics['avg_move_times']

    # We'll plot them as side-by-side bars
    bar_width = 0.35

    plt.figure(figsize=(8,5))
    plt.bar(x - bar_width/2, evals_vals, width=bar_width, color='skyblue', label='avg_stockfish_evals')
    plt.bar(x + bar_width/2, times_vals, width=bar_width, color='orange', label='avg_move_times')

    plt.title('Implementation Comparison: Stockfish Eval & Move Times')
    plt.xlabel('Implementation')
    plt.ylabel('Value')
    plt.xticks(x, x_labels)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # (B) CHART: total non-zero blunders
    # ------------------------------
    # If 'blunders' is a list of numeric values per game, 
    # we count how many are non-zero in each row.
    def count_nonzero_blunders(arr):
        if isinstance(arr, list):
            return sum(1 for x in arr if x != 0)
        return 0

    sub_df['blunder_count'] = sub_df['blunders'].apply(count_nonzero_blunders)

    # Group by implementation, sum or mean the counts
    # sum -> total number of non-zero blunders across all games in that group
    # mean -> average non-zero blunders per game in that group
    blunder_sums = sub_df.groupby('implementation')['blunder_count'].sum()

    # Sort index for consistent order
    blunder_sums = blunder_sums.sort_index()

    # Plot
    plt.figure(figsize=(6,4))
    x_labels = blunder_sums.index.tolist()
    x = np.arange(len(x_labels))
    plt.bar(x, blunder_sums, color='red', alpha=0.7)
    plt.title('Total Number of Non-zero Blunders by Implementation')
    plt.xlabel('Implementation')
    plt.ylabel('Total Non-zero Blunders')
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()

def plot_avg_blunders_whole_phase(df):
    #Happy
    sub_df = df[
        (df['phase'] == 'whole') &
        (df['implementation'].isin(['substitution', 'integration']))
    ].copy()

    def count_nonzero_blunders(arr):
        if isinstance(arr, list):
            return sum(1 for x in arr if x != 0)
        return 0

    sub_df['blunder_count'] = sub_df['blunders'].apply(count_nonzero_blunders)

    group = sub_df.groupby('implementation')['blunder_count'].mean().sort_index()

    # Plot
    plt.figure(figsize=(6,4))
    x_labels = group.index.tolist()
    x = np.arange(len(x_labels))

    plt.bar(x, group.values, color='red', alpha=0.7)
    plt.title("Average Number of Blunders per game")
    plt.xlabel('Implementation')
    plt.ylabel('Average Blunders')
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()

def plot_implementation_stockfish_eval(df):
    """
    Plots the average Stockfish evaluation for each implementation (substitution vs. integration).
    """
    # Filter for relevant implementations if needed
    sub_df = df[df['implementation'].isin(['substitution', 'integration'])].copy()
    
    # Group by 'implementation' and compute the mean of avg_stockfish_evals
    group = sub_df.groupby('implementation')['avg_stockfish_evals'].mean().sort_index()
    
    # Plot the grouped data as a bar chart
    x_labels = group.index.tolist()
    x = range(len(x_labels))
    
    plt.figure(figsize=(6,4))
    plt.bar(x, group.values, color='skyblue', alpha=0.8)
    plt.xlabel('Implementation')
    plt.ylabel('Average Stockfish Evaluation')
    plt.title('Average Stockfish Evaluation by Implementation')
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()

def plot_impl_vs_avg_blunder_value_whole(df):
    """
    Filters the DataFrame for games where phase is 'whole'.
    For each game, computes the average value of blunders (ignoring 0 values).
    Groups by implementation and plots the average nonzero blunder value for each.
    """
    # Filter rows for phase 'whole' and valid implementations
    filtered_df = df[(df['phase'] == 'whole') & 
                     (df['implementation'].isin(['substitution', 'integration']))].copy()
    
    # Function to compute average of nonzero values from a list
    def avg_nonzero(vals):
        if isinstance(vals, list):
            nonzero = [x for x in vals if x != 0]
            if nonzero:
                return np.mean(nonzero)
        return np.nan
    
    # Create a new column with the average of nonzero blunders for each game
    filtered_df['avg_nonzero_blunders'] = filtered_df['blunders'].apply(avg_nonzero)
    
    # Group by implementation and compute the mean of avg_nonzero_blunders
    group_avg = filtered_df.groupby('implementation')['avg_nonzero_blunders'].mean().sort_index()
    
    # Plot the results as a bar chart
    x_labels = group_avg.index.tolist()
    x = range(len(x_labels))
    
    plt.figure(figsize=(6,4))
    plt.bar(x, group_avg.values, color='green', alpha=0.7)
    plt.xlabel('Implementation')
    plt.ylabel('Average Non-Zero Blunder Value')
    plt.title("Implementation vs. Average Non-Zero Blunder Value (Whole Games)")
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()


def plot_combined_implementation_eval_blunders(df):
    """
    For games where phase == 'whole', this function computes:
      - The average Stockfish evaluation for each implementation.
      - The average non-zero blunder value for each implementation,
        where for each game we compute the mean of the values in the 'blunders' list that are not 0.
    It then plots a combined bar chart with two y-axes:
      - Left y-axis: avg_stockfish_evals.
      - Right y-axis: avg non-zero blunders.
    """
    # Filter for games with phase 'whole' and valid implementations
    sub_df = df[(df['phase'] == 'whole') & 
                (df['implementation'].isin(['substitution', 'integration']))].copy()
    
    # Group average for Stockfish evaluation
    group_evals = sub_df.groupby('implementation')['avg_stockfish_evals'].mean().sort_index()
    
    # Function to compute average of non-zero blunders in a list
    def avg_nonzero(vals):
        if isinstance(vals, list):
            nonzero = [x for x in vals if x != 0]
            if nonzero:
                return np.mean(nonzero)
        return np.nan
    
    # Create a new column with average non-zero blunder value for each game
    sub_df['avg_nonzero_blunders'] = sub_df['blunders'].apply(avg_nonzero)
    group_blunders = sub_df.groupby('implementation')['avg_nonzero_blunders'].mean().sort_index()
    
    # Prepare x positions for each implementation
    x_labels = group_evals.index.tolist()
    x = np.arange(len(x_labels))
    bar_width = 0.35

    # Create the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(8,5))
    
    # Plot avg_stockfish_evals as bars on the left axis
    bars1 = ax1.bar(x - bar_width/2, group_evals.values, width=bar_width, color='skyblue', label='Avg Stockfish Eval')
    ax1.set_ylabel('Avg Stockfish Evaluation')
    ax1.set_xlabel('Implementation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    
    # Create a second y-axis for avg_nonzero_blunders
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + bar_width/2, group_blunders.values, width=bar_width, color='green', label='Avg Non-Zero Blunder')
    ax2.set_ylabel('Avg Blunder Value')
    
    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='best')
    
    plt.title('Implementation vs. Avg Stockfish Eval and Avg Blunders ')
    plt.tight_layout()
    plt.show()


def plot_combined_same_scale(df):
    #Happy
    """
    Filters the DataFrame for games where phase is 'whole' and implementation is either
    'substitution' or 'integration'. It computes:
      - The average Stockfish evaluation (avg_stockfish_evals) for each implementation.
      - The average of nonzero blunders for each game, then computes the mean per implementation.
    It then plots a grouped bar chart (using the same y-axis) for both metrics.
    """
    # Filter for phase 'whole' and valid implementations
    sub_df = df[(df['phase'] == 'whole') &
                (df['implementation'].isin(['substitution', 'integration']))].copy()

    # Group average for Stockfish evaluation
    group_evals = sub_df.groupby('implementation')['avg_stockfish_evals'].mean().sort_index()
    
    # Function to compute average of nonzero blunders from a list
    def avg_nonzero(vals):
        if isinstance(vals, list):
            nonzero = [x for x in vals if x != 0]
            if nonzero:
                return np.mean(nonzero)
        return np.nan
    
    # Create a new column with the average of nonzero blunders for each game
    sub_df['avg_nonzero_blunders'] = sub_df['blunders'].apply(avg_nonzero)
    group_blunders = sub_df.groupby('implementation')['avg_nonzero_blunders'].mean().sort_index()
    
    # Prepare data for grouped bar chart
    implementations = group_evals.index.tolist()
    x = np.arange(len(implementations))
    bar_width = 0.35

    plt.figure(figsize=(8,5))
    
    # Plot both sets of bars on the same axis
    plt.bar(x - bar_width/2, group_evals.values, width=bar_width, color='skyblue', label='Avg Stockfish Eval')
    plt.bar(x + bar_width/2, group_blunders.values, width=bar_width, color='green', label='Avg Blunder value')
    
    plt.xlabel('Implementation')
    plt.ylabel('Value')
    plt.title("Implementation vs. Stockfish Eval and Blunder Values")
    plt.xticks(x, implementations)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_avg_nonzero_blunders_per_game(df):
    """
    Filters the DataFrame for games where phase is 'whole'.
    For each game, counts the number of non-zero values in the 'blunders' list.
    Groups by implementation and plots the average non-zero blunder count per game.
    """
    # Filter for games in the whole phase and for valid implementations
    sub_df = df[(df['phase'] == 'whole') & 
                (df['implementation'].isin(['substitution', 'integration']))].copy()
    
    # Helper: Count non-zero blunders in each game's blunders list
    def count_nonzero(vals):
        if isinstance(vals, list):
            return sum(1 for x in vals if x != 0)
        return 0
    
    # Create a new column with the count of non-zero blunders for each game
    sub_df['nonzero_blunder_count'] = sub_df['blunders'].apply(count_nonzero)
    
    # Group by implementation and compute the average nonzero blunder count per game
    group = sub_df.groupby('implementation')['nonzero_blunder_count'].mean().sort_index()
    
    # Prepare x-axis positions
    x_labels = group.index.tolist()
    x = np.arange(len(x_labels))
    
    # Plot the result as a bar chart
    plt.figure(figsize=(6,4))
    plt.bar(x, group.values, color='orange', alpha=0.7)
    plt.xlabel('Implementation')
    plt.ylabel('Average Non-Zero Blunders per Game')
    plt.title("Average Non-Zero Blunders per Game (Whole Games)")
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()


def safe_mean_conf(x):
    """
    Compute the mean of values in list x that are finite (i.e. not inf, -inf, or nan).
    If no finite value exists, return np.nan.
    """
    if isinstance(x, list) and len(x) > 0:
        # Filter out non-finite values using np.isfinite
        finite_vals = [float(val) for val in x if np.isfinite(float(val))]
        if finite_vals:
            return np.mean(finite_vals)
    return np.nan


def plot_avg_confidence_whole_phase(df):
    """
    Filters the DataFrame for games where phase is 'whole' and implementation is either
    'substitution' or 'integration'. For each game, computes the average value from the
    'confidences' list, ignoring infinite or non-finite values.
    Then groups by implementation and plots the average confidence.
    """
    # Filter for whole game phase and valid implementations
    sub_df = df[(df['phase'] == 'whole') &
                (df['implementation'].isin(['substitution', 'integration']))].copy()
    
    # Compute the mean confidence per game while filtering out inf, -inf, and NaN values.
    sub_df['mean_confidence'] = sub_df['confidences'].apply(safe_mean_conf)
    
    # Group by implementation and compute the mean confidence
    group_conf = sub_df.groupby('implementation')['mean_confidence'].mean().sort_index()

    print("Average confidence per implementation (finite values only):")
    print(group_conf)
    
    # Prepare the bar chart
    x_labels = group_conf.index.tolist()
    x = range(len(x_labels))
    
    plt.figure(figsize=(6,4))
    plt.bar(x, group_conf.values, color='purple', alpha=0.75)
    plt.xlabel('Implementation')
    plt.ylabel('Average Confidence')
    plt.title("Average Confidence per Game (Whole Games)")
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()

def plot_avg_mobility_whole_phase(df):
    """
    Filters the DataFrame for games with phase 'whole' and implementation in 
    ['substitution', 'integration']. For each game, computes the average of 
    the values from the 'mobilities' list (including zeros).
    Then groups by implementation and plots the average mobility per game.
    """
    sub_df = df[(df['phase'] == 'whole') & 
                (df['implementation'].isin(['substitution', 'integration']))].copy()
    
    # Helper: compute the mean of values in the list (including zeros)
    def avg_all(vals):
        if isinstance(vals, list) and len(vals) > 0:
            return np.mean(vals)
        return np.nan
    
    sub_df['avg_mobilities_all'] = sub_df['mobilities'].apply(avg_all)
    
    # Group by implementation and compute the mean of avg_mobilities_all
    group_mob = sub_df.groupby('implementation')['avg_mobilities_all'].mean().sort_index()
    
    # Plot the result as a bar chart
    x_labels = group_mob.index.tolist()
    x = range(len(x_labels))
    plt.figure(figsize=(6,4))
    plt.bar(x, group_mob.values, color='teal', alpha=0.75)
    plt.xlabel('Implementation')
    plt.ylabel('Average Mobility per Game')
    plt.title("Average Mobility per Game")
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()

def plot_avg_good_moves_whole_phase(df):
    """
    Filters the DataFrame for games with phase 'whole' and implementation in 
    ['substitution', 'integration']. For each game, computes the average of 
    nonzero values from the 'good_moves' list (ignoring 0 values).
    Then groups by implementation and plots the average nonzero good moves per game.
    """
    sub_df = df[(df['phase'] == 'whole') &
                (df['implementation'].isin(['substitution', 'integration']))].copy()
    
    # Helper: compute average of nonzero values from a list
    def avg_nonzero(vals):
        if isinstance(vals, list) and len(vals) > 0:
            nonzero = [x for x in vals if x != 0]
            if nonzero:
                return np.mean(nonzero)
        return np.nan

    sub_df['avg_nonzero_good_moves'] = sub_df['good_moves'].apply(avg_nonzero)
    
    group_good = sub_df.groupby('implementation')['avg_nonzero_good_moves'].mean().sort_index()
    
    x_labels = group_good.index.tolist()
    x = range(len(x_labels))
    plt.figure(figsize=(6,4))
    plt.bar(x, group_good.values, color='darkorange', alpha=0.75)
    plt.xlabel('Implementation')
    plt.ylabel('Average Good Moves per Game')
    plt.title("Average  Good Moves per Game")
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()


def plot_win_rate_nb_weight(df):
    """
    For rows where implementation is 'integration', computes the win rate (percentage of wins)
    for each nb_weight value. A win is defined as result == "1-0".
    
    Adjust the win condition if needed (e.g., if your engine's color changes the winning result).
    """
    # Filter for integration implementation and ensure nb_weight is valid.
    sub_df = df[(df['implementation'] == 'integration') & (df['nb_weight'].notna())].copy()
    
    # Define win: here, we assume a win is represented by the result "1-0".
    # If your engine can be black, adjust this condition accordingly.
    sub_df['win'] = sub_df['result'].apply(lambda x: 1 if x == "1-0" else 0)
    
    # Group by nb_weight and compute the win rate (mean of win values)
    win_rate = sub_df.groupby('nb_weight')['win'].mean().sort_index()
    
    # Plot the win rate as a bar chart
    x_labels = win_rate.index.tolist()
    x = range(len(x_labels))
    
    plt.figure(figsize=(6,4))
    plt.bar(x, win_rate.values, color='blue', alpha=0.8)
    plt.xlabel('nb_weight')
    plt.ylabel('Win Rate')
    plt.title('Win Rate by nb_weight (Integration Implementation)')
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()

def plot_win_rate_nb_weight_stockfish(df):
    """
    Filters the DataFrame for games where:
      - Opponent is 'stockfish'
      - Implementation is 'integration' (since nb_weight is variable only for integration)
    Then, for each game, it computes a binary win indicator (assumes win is "1-0").
    Groups the data by nb_weight and computes the win rate for each group.
    Plots the win rate (as a percentage) against nb_weight.
    """
    # Filter for games where opponent is stockfish and implementation is integration
    sub_df = df[(df['opponent'] == 'stockfish') & (df['implementation'] == 'integration')].copy()
    
    # Ensure nb_weight is numeric (and not NaN)
    sub_df['nb_weight'] = pd.to_numeric(sub_df['nb_weight'], errors='coerce')
    sub_df = sub_df[sub_df['nb_weight'].notna()]
    
    # Define win indicator: here we assume "1-0" is a win.
    # (Adjust this condition if your engine sometimes plays black.)
    sub_df['win'] = sub_df['result'].apply(lambda x: 1 if x == "1-0" else 0)
    
    # Group by nb_weight and compute the win rate (mean of win indicator)
    win_rate = sub_df.groupby('nb_weight')['win'].mean().sort_index()
    
    # Convert win rate to percentage (optional)
    win_rate_percent = win_rate * 100
    
    # Plot the results as a bar chart
    x_labels = win_rate_percent.index.tolist()
    x = range(len(x_labels))
    
    plt.figure(figsize=(6,4))
    plt.bar(x, win_rate_percent.values, color='blue', alpha=0.8)
    plt.xlabel('nb_weight')
    plt.ylabel('Win Rate (%)')
    plt.title('Win Rate against Stockfish by nb_weight (Integration)')
    plt.xticks(x, x_labels)
    plt.tight_layout()
    plt.show()

def count_nonzero(vals):
    """Count the number of nonzero values in a list."""
    if isinstance(vals, list) and len(vals) > 0:
        return sum(1 for x in vals if x != 0)
    return np.nan
    
def compute_group_metrics(df):
    """
    For games where phase is 'whole' and implementation is either 'substitution' or 'integration',
    compute the following metrics (per game):
      1. Average move time (from avg_move_times)
      2. Average nodes explored (from avg_nodes_explored)
      3. Average Stockfish evaluation (from avg_stockfish_evals)
      4. Average nonzero blunders per game (count nonzero elements in 'blunders')
      5. Average nonzero good moves per game (count nonzero elements in 'good_moves')
      6. Average mobility (from mobilities, including zeros)
      
    Then group by implementation and compute the mean of each metric.
    """
    sub_df = df[(df['phase'] == 'whole') & (df['implementation'].isin(['substitution','integration']))].copy()
    
    # Compute nonzero counts for blunders and good_moves
    sub_df['nonzero_blunder_count'] = sub_df['blunders'].apply(count_nonzero)
    sub_df['nonzero_good_moves'] = sub_df['good_moves'].apply(count_nonzero)
    
    # Compute average mobility (including zeros) from mobilities
    sub_df['mean_mobility'] = sub_df['mobilities'].apply(safe_mean)
    
    # Ensure the direct numeric columns are numeric
    for col in ['avg_move_times', 'avg_nodes_explored', 'avg_stockfish_evals']:
        sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
    
    # Group by implementation and compute means
    group_metrics = sub_df.groupby('implementation').agg({
        'avg_move_times': 'mean',
        'avg_nodes_explored': 'mean',
        'avg_stockfish_evals': 'mean',
        'nonzero_blunder_count': 'mean',
        'nonzero_good_moves': 'mean',
        'mean_mobility': 'mean'
    })
    
    # Rename columns for clarity in the radar chart
    group_metrics = group_metrics.rename(columns={
        'avg_move_times': 'Avg Move Time',
        'avg_nodes_explored': 'Avg Nodes Explored',
        'avg_stockfish_evals': 'Avg Stockfish Evals',
        'nonzero_blunder_count': 'Avg Nonzero Blunders',
        'nonzero_good_moves': 'Avg Nonzero Good Moves',
        'mean_mobility': 'Avg Mobility'
    })
    
    # Ensure that each cell is a scalar float
    group_metrics = group_metrics.apply(pd.to_numeric, errors='coerce')
    
    return group_metrics

# --- Radar Chart Plotting ---

def plot_radar_chart(group_metrics):


    """
    Plots a radar chart comparing the following metrics between implementations:
      - Avg Move Time
      - Avg Nodes Explored
      - Avg Stockfish Evals
      - Avg Nonzero Blunders
      - Avg Nonzero Good Moves
      - Avg Mobility
    """
    # List of metrics (categories)
    categories = list(group_metrics.columns)
    N = len(categories)
    
    # Compute angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    
    # Plot data for each implementation
    for impl in group_metrics.index:
        values = group_metrics.loc[impl].tolist()
        # Check that values are floats (or convert them)
        values = [float(v) if v is not None else 0 for v in values]
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, linewidth=2, label=impl)
        ax.fill(angles, values, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Radar Chart of Metrics by Implementation", y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

def compute_feature_set_metrics(df):
    """
    For games where phase is 'whole', group by feature_selection and compute:
      - Avg Move Time (from avg_move_times)
      - Win Rate (1 if result is "1-0", else 0)
      - Avg Nonzero Blunders per Game (ignoring 0 values in 'blunders')
      - Avg Mobility (mean of 'mobilities', including zeros)
      - Avg Piece Balance (from avg_piece_balances)
    Returns a DataFrame with feature_selection as index and these metrics as columns.
    """
    sub_df = df[df['phase'] == 'whole'].copy()
    
    # Ensure feature_selection is treated as a string category.
    sub_df['feature_selection'] = sub_df['feature_selection'].astype(str)
    
    # Create win indicator (assumes "1-0" is a win; adjust if necessary)
    sub_df['win'] = sub_df['result'].apply(lambda x: 1 if x == "1-0" else 0)
    
    # Compute average nonzero blunders (ignoring 0 values)
    sub_df['nonzero_blunder_count'] = sub_df['blunders'].apply(count_nonzero)
    
    # Compute average mobility (including zeros)
    sub_df['mean_mobility'] = sub_df['mobilities'].apply(safe_mean)
    
    # Group by feature_selection and compute the mean for each metric.
    metrics = sub_df.groupby('feature_selection').agg({
        'win': 'mean',
        'avg_piece_balances': 'mean',
        'avg_move_times': 'mean',
        'nonzero_blunder_count': 'mean',
        'mean_mobility': 'mean'
    })
    
    # Rename columns for clarity in the plot.
    metrics = metrics.rename(columns={
        'win': 'Win Rate',
        'avg_piece_balances': 'Avg Piece Balance',
        'avg_move_times': 'Avg Move Time',
        'nonzero_blunder_count': 'Avg Blunder Value',
        'mean_mobility': 'Avg Mobility'
    })
    
    # Ensure each cell is a scalar float.
    metrics = metrics.apply(pd.to_numeric, errors='coerce')
    
    return metrics

def plot_feature_set_comparison_bar_chart(metrics):
    """
    Given a DataFrame 'metrics' with index as feature_selection (e.g., '0' and '3') 
    and columns as performance metrics:
      - Avg Move Time
      - Win Rate
      - Avg Nonzero Blunders
      - Avg Mobility
    This function plots a grouped bar chart comparing these metrics between feature sets.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Categories are the performance metrics.
    categories = list(metrics.columns)
    n_categories = len(categories)
    
    # Feature sets (e.g., "0" and "3")
    feature_sets = metrics.index.tolist()
    n_features = len(feature_sets)
    
    # Setup positions for the categories on the x-axis.
    x = np.arange(n_categories)
    bar_width = 0.35  # Adjust as needed
    
    plt.figure(figsize=(10,6))
    
    # For each feature set, plot a set of bars for all metrics.
    for i, feat in enumerate(feature_sets):
        values = metrics.loc[feat].values
        # Offset the positions: centers are shifted so that each metric gets multiple bars.
        offset = (i - n_features/2) * bar_width + bar_width/2
        plt.bar(x + offset, values, width=bar_width, label=f"Feature Set {feat}")
    
    plt.xticks(x, categories)
    plt.ylabel("Value")
    plt.title("Comparison of Performance Metrics by Feature Set")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------
# 4) MAIN SCRIPT ENTRY POINT
# -------------------------------------------------------

def main():
    df = load_and_prepare_data('chess_results.csv')

    # plot_avg_blunders_whole_phase(df)    

    # plot_implementation_stockfish_eval(df)
    # plot_impl_vs_avg_blunder_value_whole(df)

    # plot_combined_implementation_eval_blunders(df)

    # plot_combined_same_scale(df) # happy
    # plot_avg_nonzero_blunders_per_game(df) # happy

    #plot_avg_confidence_whole_phase(df) # dont use confidence

    # plot_avg_mobility_whole_phase(df) # happy
    # plot_avg_good_moves_whole_phase(df) # happy

    # plot_win_rate_nb_weight(df) # below agaisnt stockfish is better
    # plot_win_rate_nb_weight_stockfish(df) # happy

    # plot_radar_chart(compute_group_metrics(df)) #Doesnt look useful

    #plot_opponent_comparison(df) # happy

    # plot_feature_set_comparison_bar_chart(compute_feature_set_metrics(df))   # happy




    # # 1. Implementation comparison (Substitution vs Integration)
    # plot_implementation_comparison(df)
    
    # # 2. NB weight comparison (only for integration)
    # plot_nb_weight_comparison(df)
    
    # # 3. Opponent comparison (Random vs Stockfish)
    # plot_opponent_comparison(df)
    
    # # 4. Feature selection comparison (0 vs 3)
    #plot_feature_selection_comparison(df)
    
    # # 5. Dataset comparison (master vs beginner vs random)
    plot_dataset_comparison(df)
    
    # # 6. Phase comparison (whole, opening, midgame, endgame)
    # plot_phase_comparison(df)
    
    # # 7. Colour comparison (white vs black)
    # plot_colour_comparison(df)
    
    # # 8. Correlation matrix
    # plot_correlation_matrix(df)
    
    # # 9. Time evolution of moves (example game)
    # plot_time_move_evolution(df)
    
    # # 10. Confidence usage
    # plot_confidence_usage(df)
    
    # # 11. Nodes vs. performance scatter plot
    # plot_nodes_vs_performance(df)

if __name__ == '__main__':
    main()

