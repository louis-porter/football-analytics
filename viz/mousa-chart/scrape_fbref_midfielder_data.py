import pandas as pd


def scrape_retention_metrics():
    df_possession = pd.read_html("https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats",
                    attrs={"id": "stats_possession"})[0]

    df_retention = df_possession[[
        ('Unnamed: 1_level_0', 'Player'),
        ('Unnamed: 4_level_0', 'Squad'),
        ('Unnamed: 3_level_0', 'Pos'),
        ('Unnamed: 8_level_0', '90s'),
        ('Take-Ons', 'Succ%'),
        ('Carries', 'Mis'),
        ('Carries', 'Dis')
    ]]



    df_misc = pd.read_html("https://fbref.com/en/comps/Big5/misc/players/Big-5-European-Leagues-Stats",
                    attrs={"id": "stats_misc"})[0]

    df_misc = df_misc[[('Unnamed: 1_level_0', 'Player'), ('Performance', 'Fld')]]

    df_retention = df_retention.merge(
        df_misc,
        left_on=[('Unnamed: 1_level_0', 'Player')],
        right_on=[('Unnamed: 1_level_0', 'Player')],
        how='left'
    )



    df_passing = pd.read_html("https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats",
                    attrs={"id": "stats_passing"})[0]


    df_pass_cmp = df_passing[[('Unnamed: 1_level_0', 'Player'), ('Total', 'Cmp%')]]

    df_retention = df_retention.merge(
        df_pass_cmp,
        left_on=[('Unnamed: 1_level_0', 'Player')],
        right_on=[('Unnamed: 1_level_0', 'Player')],
        how='left'
    )

    df_retention.columns = [
        "player",
        "team", 
        "position",
        "90s",
        "dribble_success_rate",
        "miscontrols",
        "dispossessed",
        "fouls_drawn",
        "pass_completion_rate"
    ]

    metrics_to_adjust = ['miscontrols', 'dispossessed', 'fouls_drawn', 'pass_completion_rate']
    df_retention['90s'] = pd.to_numeric(df_retention['90s'], errors='coerce')

    for metric in metrics_to_adjust:
            df_retention[metric] = pd.to_numeric(df_retention[metric], errors='coerce')
            df_retention[metric] = df_retention[metric] / df_retention['90s']

    return df_retention


def scrape_progression_metrics():

    df_passing = pd.read_html("https://fbref.com/en/comps/Big5/passing/players/Big-5-European-Leagues-Stats",
                        attrs={"id": "stats_passing"})[0]

    df_progression = df_passing[[
        ('Unnamed: 1_level_0', 'Player'),
        ('Unnamed: 4_level_0', 'Squad'),
        ('Unnamed: 3_level_0', 'Pos'),
        ('Unnamed: 8_level_0', '90s'),
        ('Total', 'PrgDist'),
        ('Unnamed: 31_level_0', 'PrgP'),
        ('Unnamed: 28_level_0', '1/3')
    ]]

    df_possession = pd.read_html("https://fbref.com/en/comps/Big5/possession/players/Big-5-European-Leagues-Stats",
                        attrs={"id": "stats_possession"})[0]

    df_possession = df_possession[[('Unnamed: 1_level_0', 'Player'), ('Carries', '1/3'), ('Carries', 'PrgDist'), ('Carries', 'PrgC')]]

    df_progression = df_progression.merge(
            df_possession,
            left_on=[('Unnamed: 1_level_0', 'Player')],
            right_on=[('Unnamed: 1_level_0', 'Player')],
            how='left'
        )
    
    df_progression.columns = [
        "player",
        "team", 
        "position",
        "90s",
        "pass_prog_dist",
        "prog_passes",
        "passes_final_third",
        "carries_final_third",
        "carries_prog_dist",
        "prog_carries"
    ]

    metrics_to_adjust = ['pass_prog_dist', 'prog_passes', 'passes_final_third', 'carries_final_third', 'carries_prog_dist', 'prog_carries']
    df_progression['90s'] = pd.to_numeric(df_progression['90s'], errors='coerce')

    for metric in metrics_to_adjust:
            df_progression[metric] = pd.to_numeric(df_progression[metric], errors='coerce')
            df_progression[metric] = df_progression[metric] / df_progression['90s']

    return df_progression


def scrape_def_actions_metrics():

    df_def_act = pd.read_html("https://fbref.com/en/comps/Big5/defense/players/Big-5-European-Leagues-Stats",
                            attrs={"id": "stats_defense"})[0]

    df_ball_winning = df_def_act[[
        ('Unnamed: 1_level_0', 'Player'),
        ('Unnamed: 4_level_0', 'Squad'),
        ('Unnamed: 3_level_0', 'Pos'),
        ('Unnamed: 8_level_0', '90s'),
        ('Tackles', 'Tkl'),
        ('Unnamed: 21_level_0', 'Int'),
        ('Challenges', 'Tkl%'),
        ('Unnamed: 24_level_0', 'Err'),
    ]]

    df_misc = pd.read_html("https://fbref.com/en/comps/Big5/misc/players/Big-5-European-Leagues-Stats",
                        attrs={"id": "stats_misc"})[0]

    df_misc = df_misc[[('Unnamed: 1_level_0', 'Player'), ('Performance', 'Fls'), ('Performance', 'Recov')]]

    df_ball_winning = df_ball_winning.merge(
            df_misc,
            left_on=[('Unnamed: 1_level_0', 'Player')],
            right_on=[('Unnamed: 1_level_0', 'Player')],
            how='left'
        )

    df_ball_winning.columns = [
            "player",
            "team", 
            "position",
            "90s",
            "tackles",
            "interceptions",
            "tackle_win_rate",
            "errors",
            "fouls_committed",
            "ball_recoveries"
        ]

    df_team_stats_25 = pd.read_html("https://fbref.com/en/comps/Big5/stats/squads/Big-5-European-Leagues-Stats",
                        attrs={"id": "stats_teams_standard_for"})[0]

    df_team_stats_25 = df_team_stats_25[[('Unnamed: 1_level_0', 'Squad'), ('Unnamed: 5_level_0', 'Poss')]]

    df_team_stats_24 = pd.read_html("https://fbref.com/en/comps/Big5/2023-2024/stats/squads/2023-2024-Big-5-European-Leagues-Stats",
                        attrs={"id": "stats_teams_standard_for"})[0]

    df_team_stats_24 = df_team_stats_24[[('Unnamed: 1_level_0', 'Squad'), ('Unnamed: 5_level_0', 'Poss')]]

    df_team_stats_25.columns = ['Squad', 'Poss_25']
    df_team_stats_24.columns = ['Squad', 'Poss_24']

    df_team_poss = df_team_stats_25.merge(
        df_team_stats_24,
        on='Squad',
        how='outer' 
    )

    df_team_poss['Avg_Poss'] = df_team_poss[['Poss_24', 'Poss_25']].mean(axis=1)
    df_team_poss['Avg_Poss'] = df_team_poss['Avg_Poss'].round(1)
    df_team_poss_final = df_team_poss[['Squad', 'Avg_Poss']]

    df_ball_winning = df_ball_winning.merge(
        df_team_poss_final,
        left_on='team',
        right_on='Squad',
        how='left'
    )


    df_ball_winning['tackles'] = pd.to_numeric(df_ball_winning['tackles'], errors='coerce')
    df_ball_winning['interceptions'] = pd.to_numeric(df_ball_winning['interceptions'], errors='coerce')
    df_ball_winning['pos_adj_tck'] = (df_ball_winning['tackles']) * (100 / df_ball_winning['Avg_Poss'])
    df_ball_winning['pos_adj_int'] = (df_ball_winning['interceptions']) * (100 / df_ball_winning['Avg_Poss'])

    df_ball_winning = df_ball_winning.drop(['tackles', 'interceptions', 'Squad'], axis=1)

    metrics_to_adjust = ['pos_adj_tck', 'pos_adj_int', 'errors', 'fouls_committed', 'ball_recoveries']
    df_ball_winning['90s'] = pd.to_numeric(df_ball_winning['90s'], errors='coerce')

    for metric in metrics_to_adjust:
            df_ball_winning[metric] = pd.to_numeric(df_ball_winning[metric], errors='coerce')
            df_ball_winning[metric] = df_ball_winning[metric] / df_ball_winning['90s']


    return df_ball_winning

import time
df_retention = scrape_retention_metrics()
time.sleep(5)
df_progression = scrape_progression_metrics()
time.sleep(5)
df_ball_winning = scrape_def_actions_metrics()



print(f"Retention players: {len(df_retention)}, Unique players: {len(df_retention['player'].unique())}")
print(f"Progression players: {len(df_progression)}, Unique players: {len(df_progression['player'].unique())}")
print(f"Ball winning players: {len(df_ball_winning)}, Unique players: {len(df_ball_winning['player'].unique())}")

df_retention = df_retention.drop_duplicates(subset='player')
df_progression = df_progression.drop_duplicates(subset='player')
df_ball_winning = df_ball_winning.drop_duplicates(subset='player')

df = df_retention.merge(
    df_progression,
    on='player',
    how='outer'
).merge(
    df_ball_winning,
    on='player',
    how='outer'
)

cols_to_drop = [col for col in df.columns if '_x' in col or '_y' in col]
df = df.drop(cols_to_drop, axis=1)

print(df.head())

