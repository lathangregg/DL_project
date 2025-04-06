from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def combine_team_rows(df):
    """
    Combines rows from the same GAME_ID into one row (1 for away, 1 for home).
    Assumes the first team listed per GAME_ID is the away team.
    Adds 'a_' and 'h_' prefixes for features accordingly.
    """
    # Check that all games have exactly 2 rows
    game_counts = df['GAME_ID'].value_counts()
    if not (game_counts == 2).all():
        raise ValueError("Some GAME_IDs do not have exactly 2 teams.")

    # Sort the dataframe to ensure consistent ordering
    df_sorted = df.sort_values(['GAME_ID', 'TEAM_ID']).copy()

    # Columns to retain only once (shared columns)
    shared_cols = ['GAME_ID', 'GAME_SEQUENCE', 'Date', 'YEAR', 'GAME_DATE_EST']

    # Create away and home dataframes
    away_rows = df_sorted.groupby('GAME_ID').nth(0).copy()
    home_rows = df_sorted.groupby('GAME_ID').nth(1).copy()

    # Reset index so GAME_ID becomes a column again
    away_rows = away_rows.reset_index()
    home_rows = home_rows.reset_index()

    # Extract shared info from away team (can be either)
    shared = away_rows[shared_cols].copy()

    # Drop shared columns from both to avoid duplication
    away_features = away_rows.drop(columns=shared_cols)
    home_features = home_rows.drop(columns=shared_cols)

    # Add prefixes
    away_features = away_features.add_prefix('a_')
    home_features = home_features.add_prefix('h_')

    # Combine all into one DataFrame
    combined = pd.concat([shared, away_features, home_features], axis=1)

    return combined


def add_home_win_column(df):
    """
    Adds 'HOME_WIN' column (1 if home team won, 0 otherwise)
    """
    df['HOME_WIN'] = (df['h_PTS'] > df['a_PTS']).astype(int)
    return df


def split_wins_losses(df):
    """
    Converts TEAM_WINS_LOSSES strings into integer WINS and LOSSES.
    Adjusts for current game outcome to reflect record *before* the game.
    """
    for side in ['a', 'h']:
        wins_losses = df[f'{side}_TEAM_WINS_LOSSES'].str.split('-', expand=True)
        df[f'{side}_WINS'] = wins_losses[0].astype(int)
        df[f'{side}_LOSSES'] = wins_losses[1].astype(int)

        # Subtract 1 win or 1 loss depending on outcome
        if side == 'a':
            df.loc[df['HOME_WIN'] == 0, f'{side}_WINS'] -= 1  # away team won
            df.loc[df['HOME_WIN'] == 1, f'{side}_LOSSES'] -= 1  # away team lost
        else:
            df.loc[df['HOME_WIN'] == 1, f'{side}_WINS'] -= 1  # home team won
            df.loc[df['HOME_WIN'] == 0, f'{side}_LOSSES'] -= 1  # home team lost

    return df

def add_playoff_indicator(df):
    """
    Adds:
    - 'a_SEASON_GAMES_PLAYED' and 'h_SEASON_GAMES_PLAYED' as sum of wins and losses
    - 'IS_PLAYOFF_GAME' as 1 if either team has played 82+ games, else 0
    """
    df['a_SEASON_GAMES_PLAYED'] = df['a_WINS'] + df['a_LOSSES']
    df['h_SEASON_GAMES_PLAYED'] = df['h_WINS'] + df['h_LOSSES']
    
    df['IS_PLAYOFF_GAME'] = (
        (df['a_SEASON_GAMES_PLAYED'] >= 82) |
        (df['h_SEASON_GAMES_PLAYED'] >= 82)
    ).astype(int)

    return df

def combine_date_columns(df):
    """
    Combines 'Date' and 'GAME_DATE_EST' into a single cleaned 'Date' column.
    Keeps the non-null value between the two and ensures proper datetime format.
    """
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST'], errors='coerce')
    df['Date_temp'] = df['Date'].combine_first(df['GAME_DATE_EST'])
    df['Date'] = df['Date_temp']
    df.drop(columns='Date_temp', inplace=True)
    return df

def remove_all_star_and_playoff_games(df):
    """
    Removes All-Star Games and Playoff Games.
    - All-Star Games are removed using a fixed list of dates.
    - Playoff Games are removed if either team played 82 or more games.
    Assumes 'Date' column is cleaned and that 'a_WINS', 'a_LOSSES', 'h_WINS', 'h_LOSSES' are present.
    """
    all_star_dates = pd.to_datetime([
        '2012-02-26',
        '2013-02-17',
        '2014-02-16',
        '2015-02-15',
        '2016-02-14',
        '2017-02-19',
        '2018-02-18',
        '2019-02-17'
    ])
    
    # Drop rows on All-Star dates
    df = df[~df['Date'].isin(all_star_dates)]

    # Drop rows where either team has played 82+ games (playoffs)
    df['a_SEASON_GAMES_PLAYED'] = df['a_WINS'] + df['a_LOSSES']
    df['h_SEASON_GAMES_PLAYED'] = df['h_WINS'] + df['h_LOSSES']
    df = df[(df['a_SEASON_GAMES_PLAYED'] < 82) & (df['h_SEASON_GAMES_PLAYED'] < 82)]

    return df

def create_ml_dataset(df):
    # Drop detailed score and city/abbreviation columns
    drop_cols = [
        'a_PTS_QTR1','a_PTS_QTR2','a_PTS_QTR3','a_PTS_QTR4','a_PTS_OT1','a_PTS_OT2','a_PTS_OT3','a_PTS_OT4',
        'a_PTS_OT5','a_PTS_OT6','a_PTS_OT7','a_PTS_OT8','a_PTS_OT9','a_PTS_OT10','a_TEAM_ABBREVIATION','a_TEAM_CITY_NAME',
        'h_PTS_QTR1','h_PTS_QTR2','h_PTS_QTR3','h_PTS_QTR4','h_PTS_OT1','h_PTS_OT2','h_PTS_OT3','h_PTS_OT4',
        'h_PTS_OT5','h_PTS_OT6','h_PTS_OT7','h_PTS_OT8','h_PTS_OT9','h_PTS_OT10','h_TEAM_ABBREVIATION','h_TEAM_CITY_NAME'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Sort for lag calculation
    df = df.sort_values(by=['YEAR', 'Date'])

    # Initialize dictionaries to hold cumulative stats
    lag_stats = ['PTS', 'AST', 'REB', 'TOV']
    avg_stats = ['FG_PCT', 'FT_PCT', 'FG3_PCT']
    for team_type in ['a', 'h']:
        team_id_col = f'{team_type}_TEAM_ID'
        prefix = f'{team_type}_'

        for stat in lag_stats:
            col = prefix + stat
            df[f'{prefix}season_{stat}_lag'] = (
                df.groupby([team_id_col, 'YEAR'])[col].cumsum() - df[col]
            )

        for stat in avg_stats:
            col = prefix + stat
            df[f'{prefix}season_{stat}_avg'] = (
                df.groupby([team_id_col, 'YEAR'])[col].expanding().mean().shift().reset_index(level=[0,1], drop=True)
            )

        # Winning %
        wins_col = f'{team_type}_WINS'
        losses_col = f'{team_type}_LOSSES'
        games_played = df[wins_col] + df[losses_col]
        df[f'{prefix}WIN_PCT'] = df[wins_col] / games_played.replace(0, np.nan)

        # Winning % in last 10 games
        df[f'{prefix}WIN_PCT_LAST10'] = 0.5  # Default placeholder
        team_games = df.groupby(team_id_col).apply(lambda group: group.sort_values('Date')).reset_index(drop=True)

        last10_pct = []
        for idx, row in df.iterrows():
            team_id = row[team_id_col]
            game_date = row['Date']
            team_rows = df[(df[team_id_col] == team_id) & (df['Date'] < game_date)]
            last_10 = team_rows.tail(10)
            wins = 0
            if team_type == 'a':
                wins = last_10['HOME_WIN'].apply(lambda x: 1 - x).sum()
            else:
                wins = last_10['HOME_WIN'].sum()
            total = len(last_10)
            pct = wins / total if total > 0 else row[f'{prefix}WIN_PCT']
            last10_pct.append(pct)

        df[f'{prefix}WIN_PCT_LAST10'] = last10_pct

    return df

def standardize_features(df, feature_cols):
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler

def compute_relative_features(df, feature_cols):
    for col in feature_cols:
        df[col + "_rel"] = df[col] - df.groupby("season")[col].transform("mean")
    return df

def compute_team_diff(df, team1_prefix, team2_prefix, feature_cols):
    diff_features = {}
    for col in feature_cols:
        diff_features[col + "_diff"] = df[f"{team1_prefix}_{col}"] - df[f"{team2_prefix}_{col}"]
    return pd.DataFrame(diff_features)



