import pandas as pd
import numpy as np
import ast

# Load results CSV file
df = pd.read_csv("chess_results.csv")

# Function to calculate win/draw/loss percentages and counts
def calculate_win_rates(data, color_column='colour'):
    # Determine if we're playing as white or black
    if color_column in data.columns:
        # Handle results based on color played
        conditions = [
            # Win conditions
            ((data[color_column] == 'white') & (data['result'] == '1-0')) | 
            ((data[color_column] == 'black') & (data['result'] == '0-1')),
            # Draw conditions
            (data['result'] == '1/2-1/2'),
            # Loss conditions
            ((data[color_column] == 'white') & (data['result'] == '0-1')) | 
            ((data[color_column] == 'black') & (data['result'] == '1-0'))
        ]
        choices = ['win', 'draw', 'loss']
    else:
        # Simple calculation if color information unavailable
        conditions = [
            (data['result'] == '1-0'), 
            (data['result'] == '1/2-1/2'), 
            (data['result'] == '0-1')
        ]
        choices = ['win', 'draw', 'loss']
    
    data['outcome'] = np.select(conditions, choices, default='unknown')
    
    # Calculate counts
    result_counts = data['outcome'].value_counts()
    total = result_counts.sum()
    
    # Calculate percentages
    percentages = (result_counts / total * 100).round(1)
    
    # Return both counts and percentages
    return {
        'counts': result_counts.to_dict(),
        'percentages': percentages.to_dict(),
        'total': total
    }

def calculate_win_rates_by_implementation_and_opponent():


    # 1. Overall comparison by implementation
    implementations = df['implementation'].unique()
    results_by_implementation = {}
    for impl in implementations:
        impl_data = df[df['implementation'] == impl]
        results_by_implementation[impl] = calculate_win_rates(impl_data)

    # 2. Breakdown by implementation and opponent
    opponents = df['opponent'].unique()
    results_by_impl_opponent = {}
    for impl in implementations:
        results_by_impl_opponent[impl] = {}
        for opp in opponents:
            filtered_data = df[(df['implementation'] == impl) & (df['opponent'] == opp)]
            results_by_impl_opponent[impl][opp] = calculate_win_rates(filtered_data)

    # Print overall results by implementation
    print("=" * 70)
    print("OVERALL WIN/DRAW/LOSS RATES BY IMPLEMENTATION")
    print("=" * 70)
    for impl in implementations:
        result = results_by_implementation[impl]
        counts = result['counts']
        percentages = result['percentages']
        total = result['total']
        
        win_count = counts.get('win', 0)
        draw_count = counts.get('draw', 0)
        loss_count = counts.get('loss', 0)
        
        win_pct = percentages.get('win', 0)
        draw_pct = percentages.get('draw', 0)
        loss_pct = percentages.get('loss', 0)
        
        print(f"{impl} (Total games: {total}):")
        print(f"  Win:  {win_count} games ({win_pct}%)")
        print(f"  Draw: {draw_count} games ({draw_pct}%)")
        print(f"  Loss: {loss_count} games ({loss_pct}%)")
        print()

    # Print results broken down by implementation and opponent
    print("\n" + "=" * 70)
    print("WIN/DRAW/LOSS RATES BY IMPLEMENTATION AND OPPONENT")
    print("=" * 70)
    for opp in opponents:
        print(f"\nOpponent: {opp}")
        print("-" * 50)
        for impl in implementations:
            result = results_by_impl_opponent[impl][opp]
            counts = result['counts']
            percentages = result['percentages']
            total = result['total']
            
            win_count = counts.get('win', 0)
            draw_count = counts.get('draw', 0)
            loss_count = counts.get('loss', 0)
            
            win_pct = percentages.get('win', 0)
            draw_pct = percentages.get('draw', 0)
            loss_pct = percentages.get('loss', 0)
            
            print(f"{impl} (Total games: {total}):")
            print(f"  Win:  {win_count} games ({win_pct}%)")
            print(f"  Draw: {draw_count} games ({draw_pct}%)")
            print(f"  Loss: {loss_count} games ({loss_pct}%)")
            print()

def calculate_average_stockfish_evaluation():
    print("\n" + "=" * 70)
    print("AVERAGE STOCKFISH EVALUATION BY IMPLEMENTATION")
    print("=" * 70)
    
    # Check if stockfish evaluation column exists
    eval_column = None
    potential_columns = ['avg_stockfish_evals', 'stockfish_evals', 'avg_stockfish_eval', 'stockfish_eval']
    for col in potential_columns:
        if col in df.columns:
            eval_column = col
            break
    
    if eval_column is None:
        print("No Stockfish evaluation column found in the dataset")
        print("Available columns:", df.columns.tolist())
        return
    
    # Get average evaluation by implementation
    implementations = df['implementation'].unique()
    for impl in implementations:
        impl_data = df[df['implementation'] == impl]
        avg_eval = impl_data[eval_column].mean()
        
        print(f"{impl}: {avg_eval:.2f}")
        
        # Breakdown by opponent
        print("\n  By opponent:")
        opponents = df['opponent'].unique()
        for opp in opponents:
            filtered_data = df[(df['implementation'] == impl) & (df['opponent'] == opp)]
            if not filtered_data.empty:
                avg_eval_by_opp = filtered_data[eval_column].mean()
                print(f"    vs {opp}: {avg_eval_by_opp:.2f}")
        print()

# Call the function
def calculate_average_blunders():
    print("\n" + "=" * 70)
    print("AVERAGE BLUNDER SEVERITY BY IMPLEMENTATION (ONLY NON-ZERO BLUNDERS)")
    print("=" * 70)
    
    import ast
    
    # Function to parse string representation of lists
    def parse_blunders(blunder_str):
        try:
            blunder_list = ast.literal_eval(blunder_str)
            # Filter for non-zero blunders (negative values are blunders)
            non_zero_blunders = [b for b in blunder_list if b < 0]
            return non_zero_blunders
        except:
            return []
    
    # Process all games
    implementations = df['implementation'].unique()
    for impl in implementations:
        impl_data = df[df['implementation'] == impl]
        
        # Collect all non-zero blunders for this implementation
        all_blunders = []
        for blunder_str in impl_data['blunders']:
            blunders = parse_blunders(blunder_str)
            all_blunders.extend(blunders)
        
        # Calculate average (if there are any blunders)
        if all_blunders:
            avg_blunder = sum(all_blunders) / len(all_blunders)
            print(f"{impl}: {avg_blunder:.2f} ({len(all_blunders)} blunders)")
        else:
            print(f"{impl}: No blunders found")
        
        # Breakdown by opponent
        print("\n  By opponent:")
        opponents = df['opponent'].unique()
        for opp in opponents:
            filtered_data = df[(df['implementation'] == impl) & (df['opponent'] == opp)]
            
            opp_blunders = []
            for blunder_str in filtered_data['blunders']:
                blunders = parse_blunders(blunder_str)
                opp_blunders.extend(blunders)
                
            if opp_blunders:
                avg_opp_blunder = sum(opp_blunders) / len(opp_blunders)
                print(f"    vs {opp}: {avg_opp_blunder:.2f} ({len(opp_blunders)} blunders)")
            else:
                print(f"    vs {opp}: No blunders found")
        print()

def calculate_avg_nonzero_blunders_per_game():
    print("\n" + "=" * 70)
    print("AVERAGE NUMBER OF NON-ZERO BLUNDERS PER WHOLE GAME BY IMPLEMENTATION")
    print("=" * 70)
    
    import ast
    
    # Filter for whole games if phase column exists
    if 'phase' in df.columns:
        games_df = df[df['phase'] == 'whole']
        if games_df.empty:
            print("No games with phase 'whole' found, using all games.")
            games_df = df
    else:
        games_df = df
    
    # Function to count non-zero blunders in a list
    def count_nonzero_blunders(blunder_str):
        try:
            blunder_list = ast.literal_eval(blunder_str)
            # Count non-zero blunders (negative values are blunders)
            return sum(1 for b in blunder_list if b != 0)
        except:
            return 0
    
    # Group by implementation
    implementations = games_df['implementation'].unique()
    for impl in implementations:
        impl_data = games_df[games_df['implementation'] == impl]
        
        # Calculate average non-zero blunders per game
        blunder_counts = impl_data['blunders'].apply(count_nonzero_blunders)
        avg_blunder_count = blunder_counts.mean()
        total_games = len(impl_data)
        
        print(f"{impl} ({total_games} games): {avg_blunder_count:.2f} non-zero blunders per game")
        
        # Breakdown by opponent
        print("\n  By opponent:")
        opponents = games_df['opponent'].unique()
        for opp in opponents:
            filtered_data = impl_data[impl_data['opponent'] == opp]
            
            if not filtered_data.empty:
                opp_blunder_counts = filtered_data['blunders'].apply(count_nonzero_blunders)
                avg_opp_blunder_count = opp_blunder_counts.mean()
                opp_games = len(filtered_data)
                print(f"    vs {opp} ({opp_games} games): {avg_opp_blunder_count:.2f} non-zero blunders per game")
        print()

def load_data():
    """Load and prepare the chess results data"""
    df = pd.read_csv("chess_results.csv")
    
    # Parse list columns
    list_columns = ['mobilities', 'piece_balances']
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    return df

def calculate_average_by_phase_and_implementation(df):
    """Calculate and print average mobility and piece balance by phase and implementation"""
    # Calculate average mobility per game
    def calculate_avg_metric(metric_list):
        if isinstance(metric_list, list) and len(metric_list) > 0:
            return np.mean(metric_list)
        return np.nan
    
    # Create average columns
    df['avg_game_mobility'] = df['mobilities'].apply(calculate_avg_metric)
    df['avg_game_piece_balance'] = df['piece_balances'].apply(calculate_avg_metric)
    
    # Group by phase and implementation
    mobility_grouped = df.groupby(['phase', 'implementation'])['avg_game_mobility'].mean().unstack('implementation')
    piece_balance_grouped = df.groupby(['phase', 'implementation'])['avg_game_piece_balance'].mean().unstack('implementation')
    
    # Print results
    print("\n" + "=" * 70)
    print("AVERAGE MOBILITY BY PHASE AND IMPLEMENTATION")
    print("=" * 70)
    print(mobility_grouped)
    
    print("\n\n" + "=" * 70)
    print("AVERAGE PIECE BALANCE BY PHASE AND IMPLEMENTATION")
    print("=" * 70)
    print(piece_balance_grouped)
    
    # Optional: Calculate the difference between implementations
    if len(mobility_grouped.columns) >= 2:
        print("\n\n" + "=" * 70)
        print("DIFFERENCES (integration - substitution)")
        print("=" * 70)
        
        mobility_diff = mobility_grouped['integration'] - mobility_grouped['substitution']
        piece_balance_diff = piece_balance_grouped['integration'] - piece_balance_grouped['substitution']
        
        print("\nMobility Difference:")
        print(mobility_diff)
        
        print("\nPiece Balance Difference:")
        print(piece_balance_diff)


def calculate_average_node_And_time_by_phase_and_implementation(df):
    # Define the phase order including "whole"
    phase_order = ["opening", "midgame", "endgame", "whole"]

    # Filter the DataFrame to include only rows where phase is one of the defined phases.
    df_filtered = df[df['phase'].isin(phase_order)].copy()

    # Optionally, convert 'phase' to a categorical with the defined order so the output is sorted as desired.
    df_filtered['phase'] = pd.Categorical(df_filtered['phase'], categories=phase_order, ordered=True)

    # --- Calculate Average Move Time ---
    avg_move_time = (
        df_filtered
        .groupby(['phase', 'implementation'])['avg_move_times']
        .mean()
        .reset_index()
    )
    print("Average Move Time by Phase and Implementation:")
    print(avg_move_time)

    # --- Calculate Average Nodes Explored ---
    avg_nodes_evaluated = (
        df_filtered
        .groupby(['phase', 'implementation'])['avg_nodes_explored']
        .mean()
        .reset_index()
    )
    print("Average Nodes Explored by Phase and Implementation:")
    print(avg_nodes_evaluated)


def calculate_feature_set_metrics():
    """
    Calculate and print average piece balance, non-zero blunder count,
    and mobility for each feature set.
    """
    print("\n" + "=" * 70)
    print("METRICS BY FEATURE SET")
    print("=" * 70)
    
    # Ensure we have a feature_selection column
    if 'feature_selection' not in df.columns:
        print("No 'feature_selection' column found in the dataset")
        return
        
    # Convert feature_selection to string for consistent grouping
    df['feature_selection'] = df['feature_selection'].astype(str)
    
    # Filter for whole games if phase column exists
    if 'phase' in df.columns:
        analysis_df = df[df['phase'] == 'whole'].copy()
        if analysis_df.empty:
            print("No games with phase 'whole' found, using all games.")
            analysis_df = df.copy()
    else:
        analysis_df = df.copy()
    
    # Parse and calculate piece balance average per game
    def calculate_avg_piece_balance(piece_balance_str):
        try:
            if isinstance(piece_balance_str, str):
                piece_balance_list = ast.literal_eval(piece_balance_str)
                if piece_balance_list and len(piece_balance_list) > 0:
                    return sum(piece_balance_list) / len(piece_balance_list)
            elif isinstance(piece_balance_str, list) and len(piece_balance_str) > 0:
                return sum(piece_balance_str) / len(piece_balance_str)
        except:
            pass
        return np.nan
    
    # Parse and calculate mobility average per game
    def calculate_avg_mobility(mobility_str):
        try:
            if isinstance(mobility_str, str):
                mobility_list = ast.literal_eval(mobility_str)
                if mobility_list and len(mobility_list) > 0:
                    return sum(mobility_list) / len(mobility_list)
            elif isinstance(mobility_str, list) and len(mobility_str) > 0:
                return sum(mobility_str) / len(mobility_str)
        except:
            pass
        return np.nan
    
    # Count non-zero blunders per game
    def count_nonzero_blunders(blunder_str):
        try:
            if isinstance(blunder_str, str):
                blunder_list = ast.literal_eval(blunder_str)
                return sum(1 for b in blunder_list if b != 0)
            elif isinstance(blunder_str, list):
                return sum(1 for b in blunder_str if b != 0)
        except:
            pass
        return 0
    
    # Calculate metrics for each game
    analysis_df['avg_piece_balance'] = analysis_df['piece_balances'].apply(calculate_avg_piece_balance)
    analysis_df['avg_mobility'] = analysis_df['mobilities'].apply(calculate_avg_mobility)
    analysis_df['nonzero_blunder_count'] = analysis_df['blunders'].apply(count_nonzero_blunders)
    
    # Group by feature set and calculate average metrics
    feature_set_metrics = (analysis_df
                           .groupby('feature_selection')
                           .agg({
                               'avg_piece_balance': 'mean',
                               'avg_mobility': 'mean',
                               'nonzero_blunder_count': 'mean'
                           })
                           .reset_index())
    
    # Sort by feature set (numerical order if possible)
    try:
        feature_set_metrics['feature_set_num'] = pd.to_numeric(feature_set_metrics['feature_selection'])
        feature_set_metrics = feature_set_metrics.sort_values('feature_set_num')
        feature_set_metrics = feature_set_metrics.drop('feature_set_num', axis=1)
    except:
        # If conversion fails, just sort alphabetically
        feature_set_metrics = feature_set_metrics.sort_values('feature_selection')
    
    # Print results in a nice format
    for _, row in feature_set_metrics.iterrows():
        feature_set = row['feature_selection']
        piece_balance = row['avg_piece_balance']
        mobility = row['avg_mobility']
        blunder_count = row['nonzero_blunder_count']
        
        print(f"Feature Set {feature_set}:")
        print(f"  Average Piece Balance: {piece_balance:.3f}")
        print(f"  Average Mobility: {mobility:.3f}")
        print(f"  Average Non-zero Blunders per Game: {blunder_count:.3f}")
        print()
    
    # Return the DataFrame for further analysis if needed
    return feature_set_metrics


# df = load_data()
# calculate_average_by_phase_and_implementation(df)

calculate_feature_set_metrics()
















# Call the function
