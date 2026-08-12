import streamlit as st
import pandas as pd
from data_loader import load_competitions, load_competition_events_from_api, load_player_season_stats


st.set_page_config(page_title="Statsbomb Event Data Loader", layout="wide")

# ── Password Gate ──────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.title("🔒 Access Required")
    pwd = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if pwd == st.secrets["app_password"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
# ──────────────────────────────────────────────────────────────────────────────

st.title("Statsbomb Event Data Loader")

# Load competitions data
try:
    comps_df = load_competitions()
except Exception as e:
    st.error(f"Failed to load competitions: {e}")
    st.stop()

if comps_df is not None and not comps_df.empty:
    with st.sidebar:
        st.header("Data Loading")
        # Get unique competitions with country name for distinct selection
        competitions = comps_df[['competition_id', 'competition_name', 'country_name']].drop_duplicates().copy()
        competitions['display_name'] = competitions.apply(
            lambda r: f"{r['competition_name']} ({r['country_name']})" if pd.notna(r['country_name']) and r['country_name'] else r['competition_name'],
            axis=1
        )
        competitions = competitions.sort_values('display_name')
        
        # User selects competition using display_name
        selected_comp_display = st.selectbox("Select Competition", competitions['display_name'])
        
        if selected_comp_display:
            selected_comp_row = competitions[competitions['display_name'] == selected_comp_display].iloc[0]
            selected_comp_id = selected_comp_row['competition_id']
            selected_comp_name = selected_comp_row['competition_name']
            
            # Get seasons for the selected competition
            seasons = comps_df[comps_df['competition_id'] == selected_comp_id][['season_id', 'season_name']].drop_duplicates().sort_values('season_name', ascending=False)
            
            # User selects season
            selected_season_name = st.selectbox("Select Season", seasons['season_name'])
            
            if selected_season_name:
                selected_season_id = seasons[seasons['season_name'] == selected_season_name]['season_id'].iloc[0]
                
                # Load button
                if st.button("Load Data"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("Loading filtered event data..."):
                        events_df = load_competition_events_from_api(
                            selected_comp_id, 
                            selected_season_id, 
                            progress_bar=progress_bar, 
                            status_text=status_text
                        )
                        
                        player_stats_df = load_player_season_stats(selected_comp_id, selected_season_id)
                        if not player_stats_df.empty and events_df is not None and not events_df.empty:
                            import unicodedata
                            def normalize_name(name):
                                if not isinstance(name, str):
                                    return name
                                return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
                            
                            events_df['_merge_name'] = events_df['player_name'].apply(normalize_name)
                            player_stats_df['_merge_name'] = player_stats_df['player_name'].apply(normalize_name)
                            
                            events_df = pd.merge(events_df, player_stats_df.drop(columns=['player_name']),
                                                 on=['_merge_name', 'team_name'], how='left')
                            events_df.drop(columns=['_merge_name'], inplace=True)
                    
                    # Clear progress elements
                    progress_bar.empty()
                    status_text.empty()
                        
                    if events_df is not None and not events_df.empty:
                        st.session_state['events_df'] = events_df
                        st.success(f"Data loaded successfully! Total records: {len(events_df)}")
                    else:
                        if 'events_df' in st.session_state:
                            del st.session_state['events_df']
                        st.warning("No event data found for the selected competition and season.")
                        
    # Visualization part
    if 'events_df' in st.session_state:
        df = st.session_state['events_df']
        df['player_known_name'] = df['player_known_name'].fillna(df['player_name'])
        
        st.header("Data")
        st.dataframe(df, use_container_width=True)
        
        import numpy as np
        st.header("Player Shooting Stats")
        position_options = ['CF', 'Winger/AM', 'Mid', 'FB', 'CB']
        position_filter = st.multiselect("Filter by Position", position_options, default=position_options)
        mcol1, mcol2 = st.columns(2)
        min_minutes = mcol1.number_input("Min Minutes Played", min_value=0, max_value=5000, value=1000, step=50)
        max_minutes = mcol2.number_input("Max Minutes Played", min_value=0, max_value=5000, value=5000, step=50)
        
        df_shots = df.copy()
        position_map = {
            'CF': ['Centre Forward', 'Left Centre Forward', 'Right Centre Forward', 'Secondary Striker'],
            'Winger/AM': ['Left Wing', 'Right Wing', 'Right Attacking Midfielder', 'Left Attacking Midfielder', 'Left Midfielder', 'Right Midfielder', 'Centre Attacking Midfielder'],
            'Mid': ['Centre Midfielder', 'Left Centre Midfielder', 'Right Centre Midfielder', 'Centre Defensive Midfielder', 'Left Defensive Midfielder', 'Right Defensive Midfielder'],
            'FB': ['Left Back', 'Right Back', 'Left Wing Back', 'Right Wing Back'],
            'CB': ['Left Centre Back', 'Right Centre Back', 'Centre Back']
        }
        if position_filter and 'primary_position' in df_shots.columns:
            pf = []
            for pos in position_filter:
                pf.extend(position_map.get(pos, []))
            df_shots = df_shots[df_shots['primary_position'].isin(pf)]
            
        if 'under_pressure' not in df_shots.columns:
            df_shots['under_pressure'] = False
        df_shots['under_pressure'] = df_shots['under_pressure'].fillna(False).astype(bool)
        
        dist = np.sqrt((df_shots['x'] - 120)**2 + (df_shots['y'] - 40)**2)
        df_shots['dist'] = dist
        # df_shots['zone_10'] = dist < 10
        # df_shots['zone_18'] = (dist >= 10) & (dist < 18)
        # df_shots['zone_28'] = (dist >= 18) & (dist < 28)
        # df_shots['zone_out'] = dist >= 28
        
        df_shots['is_saved'] = df_shots['outcome_name'].isin(['Saved', 'Saved to Post', 'Saved To Post'])
        df_shots['is_off_target'] = df_shots['outcome_name'].isin(['Off T', 'Saved Off T', 'Post', 'Wayward'])
        df_shots['is_blocked'] = df_shots['outcome_name'] == 'Blocked'
        df_shots['is_goal'] = df_shots['outcome_name'] == 'Goal'
        df_shots['is_post'] = df_shots['outcome_name'] == 'Post'
        
        if 'shot_statsbomb_xg' not in df_shots.columns:
            df_shots['shot_statsbomb_xg'] = 0.0
        if 'shot_statsbomb_xg2' not in df_shots.columns:
            df_shots['shot_statsbomb_xg2'] = 0.0
            
        agg_funcs = {
            'x': 'count', 
            'shot_statsbomb_xg': 'sum',
            'shot_statsbomb_xg2': 'sum',
            'is_saved': 'sum',
            'is_off_target': 'sum',
            'is_blocked': 'sum',
            'is_goal': 'sum',
            'is_post': 'sum',
            'dist': 'mean',
            # 'zone_10': 'sum',
            # 'zone_18': 'sum',
            # 'zone_28': 'sum',
            # 'zone_out': 'sum',
            'under_pressure': 'sum',
        }
        
        if 'player_known_name' in df_shots.columns:
            agg_funcs['player_known_name'] = 'first'
        if 'player_season_minutes' in df_shots.columns:
            agg_funcs['player_season_minutes'] = 'first'
        if 'primary_position' in df_shots.columns:
            agg_funcs['primary_position'] = 'first'
            
        player_agg = df_shots.groupby(['player_name', 'team_name']).agg(agg_funcs).rename(columns={
            'x': 'Total Shots', 
            'is_goal': 'Total Goals',
            'shot_statsbomb_xg': 'Total xG', 
            'shot_statsbomb_xg2': 'Total xGOT', 
            'is_saved': 'Total Shots Saved', 
            'is_off_target': 'Total Shots Off Target', 
            'is_blocked': 'Total Shots Blocked', 
            'dist': 'Avg. Shot Distance', 
            # 'zone_10': 'Shots inside 10m radius', 
            # 'zone_18': 'Shots inside 18m radius', 
            # 'zone_28': 'Shots inside 28m', 
            # 'zone_out': 'Shots outside 28m', 
            'under_pressure': 'Shots under pressure'
        })
        
        up_shots = df_shots[df_shots['under_pressure']]
        up_agg = up_shots.groupby(['player_name', 'team_name']).agg({
            'is_goal': 'sum',
            'shot_statsbomb_xg': 'sum',
            'shot_statsbomb_xg2': 'sum',
            'is_saved': 'sum',
            'is_post': 'sum'
        }).rename(columns={
            'is_goal': 'Goals under pressure', 
            'shot_statsbomb_xg': 'xG under pressure', 
            'shot_statsbomb_xg2': 'xGOT under pressure',
            'is_saved': 'Saved under pressure',
            'is_post': 'Post under pressure'
        })
        
        player_agg = player_agg.join(up_agg).fillna(0)
        
        player_agg['Total Shot Accuracy'] = (player_agg['Total Goals'] + player_agg['Total Shots Saved'] + 0.5 * player_agg['is_post']) / player_agg['Total Shots']
        player_agg['Total Shot Accuracy'] = player_agg['Total Shot Accuracy'].fillna(0.0)
        
        player_agg['Shot accuracy under pressure'] = np.where(
            player_agg['Shots under pressure'] > 0,
            (player_agg['Goals under pressure'] + player_agg['Saved under pressure'] + 0.5 * player_agg['Post under pressure']) / player_agg['Shots under pressure'],
            0.0
        )
        
        player_agg['xG Overperformance'] = player_agg['Total Goals'] - player_agg['Total xG']
        
        # Shot quality categories based on xG
        low_chance = df_shots[df_shots['shot_statsbomb_xg'] < 0.05]
        half_chance = df_shots[(df_shots['shot_statsbomb_xg'] >= 0.05) & (df_shots['shot_statsbomb_xg'] < 0.15)]
        big_chance = df_shots[df_shots['shot_statsbomb_xg'] >= 0.15]
        
        for label, subset in [('Low Chance', low_chance), ('Half Chance', half_chance), ('Big Chance', big_chance)]:
            if not subset.empty:
                chance_agg = subset.groupby(['player_name', 'team_name']).agg({'x': 'count', 'is_goal': 'sum'})
                player_agg = player_agg.join(chance_agg.rename(columns={'x': f'{label} Shots', 'is_goal': f'{label} Goals'})).fillna(0)
            else:
                player_agg[f'{label} Shots'] = 0
                player_agg[f'{label} Goals'] = 0
            player_agg[f'{label} Conversion%'] = np.where(
                player_agg[f'{label} Shots'] > 0,
                player_agg[f'{label} Goals'] / player_agg[f'{label} Shots'],
                0.0
            )
        
        def get_subset_stats(subset_df, prefix):
            if subset_df.empty:
                df_sub = pd.DataFrame(columns=[f'{prefix} Shots', f'{prefix} Shots Accuracy', f'{prefix} Conversion%'])
                df_sub.index = pd.MultiIndex.from_tuples([], names=['player_name', 'team_name'])
                return df_sub
            agg = subset_df.groupby(['player_name', 'team_name']).agg({
                'x': 'count',
                'is_goal': 'sum',
                'is_saved': 'sum',
                'is_post': 'sum'
            })
            acc = np.where(agg['x'] > 0, (agg['is_goal'] + agg['is_saved'] + 0.5 * agg['is_post']) / agg['x'], 0.0)
            conv = np.where(agg['x'] > 0, agg['is_goal'] / agg['x'], 0.0)
            return pd.DataFrame({
                f'{prefix} Shots': agg['x'],
                f'{prefix} Shots Accuracy': acc,
                f'{prefix} Conversion%': conv
            }, index=agg.index)

        subset_stats = []
        if 'body_part_name' in df_shots.columns:
            subset_stats.append(get_subset_stats(df_shots[df_shots['body_part_name'] == 'Right Foot'], 'Right Foot'))
            subset_stats.append(get_subset_stats(df_shots[df_shots['body_part_name'] == 'Left Foot'], 'Left Foot'))
            subset_stats.append(get_subset_stats(df_shots[df_shots['body_part_name'] == 'Head'], 'Headed'))
        if 'shot_first_time' in df_shots.columns:
            subset_stats.append(get_subset_stats(df_shots[df_shots['shot_first_time'] == True], 'First Touch'))
        if 'shot_one_on_one' in df_shots.columns:
            subset_stats.append(get_subset_stats(df_shots[df_shots['shot_one_on_one'] == True], 'One on One'))
        if 'shot_follows_dribble' in df_shots.columns:
            subset_stats.append(get_subset_stats(df_shots[df_shots['shot_follows_dribble'] == True], 'Dribbled'))

        new_stat_cols = []
        for stats in subset_stats:
            player_agg = player_agg.join(stats).fillna(0)
            new_stat_cols.extend(stats.columns.tolist())
        
        # Ensure correct column order
        base_cols = []
        if 'player_known_name' in player_agg.columns: base_cols.append('player_known_name')
        if 'player_season_minutes' in player_agg.columns: base_cols.append('player_season_minutes')
        if 'primary_position' in player_agg.columns: base_cols.append('primary_position')
        
        stat_cols = ['Total Shots', 'Total Goals', 'Total xG', 'Total xGOT', 'xG Overperformance', 'Total Shots Saved', 'Total Shots Off Target', 
                      'Total Shots Blocked', 'Avg. Shot Distance', 'Total Shot Accuracy', 
                      'Low Chance Shots', 'Low Chance Goals', 'Low Chance Conversion%',
                      'Half Chance Shots', 'Half Chance Goals', 'Half Chance Conversion%',
                      'Big Chance Shots', 'Big Chance Goals', 'Big Chance Conversion%',
                      # 'Shots inside 10m radius', 'Shots inside 18m radius', 'Shots inside 28m', 'Shots outside 28m', 
                      'Shots under pressure', 'Goals under pressure', 'xG under pressure', 'xGOT under pressure', 'Shot accuracy under pressure']
                      
        stat_cols.extend(new_stat_cols)
                      
        cols_order = base_cols + stat_cols
        
        if 'player_season_minutes' in player_agg.columns:
            excluded_from_p90 = ['xG Overperformance', 'Avg. Shot Distance', 'Total Shot Accuracy', 'Shot accuracy under pressure',
                                 'Low Chance Conversion%', 'Half Chance Conversion%', 'Big Chance Conversion%']
            excluded_from_p90.extend(new_stat_cols)
                    
            for col in stat_cols:
                if col not in excluded_from_p90:
                    p90_col = col + ' per90'
                    player_agg[p90_col] = np.where(
                        player_agg['player_season_minutes'] > 0,
                        (player_agg[col] / player_agg['player_season_minutes']) * 90,
                        0.0
                    )
                    cols_order.append(p90_col)
                    
        player_agg = player_agg[cols_order]
        
        if 'player_season_minutes' in player_agg.columns:
            player_agg = player_agg[(player_agg['player_season_minutes'] >= min_minutes) & (player_agg['player_season_minutes'] <= max_minutes)]
        
        st.dataframe(player_agg, use_container_width=True)
        
        st.divider()
        st.header("Dashboard Layout")
        
        col1, col2 = st.columns(2)
        teams = sorted(df['team_name'].dropna().unique().tolist()) if 'team_name' in df.columns else []
        selected_team = col1.selectbox("Select Team", teams)
        
        if selected_team:
            team_df_shots = df_shots[df_shots['team_name'] == selected_team]
            if not team_df_shots.empty:
                if 'player_known_name' in team_df_shots.columns:
                    player_map_df = team_df_shots[['player_name', 'player_known_name']].drop_duplicates()
                    name_mapping = {}
                    for _, row in player_map_df.iterrows():
                        display = row['player_known_name'] if pd.notna(row['player_known_name']) else row['player_name']
                        name_mapping[display] = row['player_name']
                else:
                    player_map_df = team_df_shots[['player_name']].drop_duplicates()
                    name_mapping = {row['player_name']: row['player_name'] for _, row in player_map_df.iterrows()}
                
                display_names = sorted(name_mapping.keys())
                selected_display_name = col2.selectbox("Select Player", display_names)
                selected_player = name_mapping.get(selected_display_name)
            else:
                selected_player = None
                col2.selectbox("Select Player", ["No players available"])
            
            if selected_player:
                player_df = df[(df['team_name'] == selected_team) & (df['player_name'] == selected_player)]
                
                if not player_df.empty and 'player_season_minutes' in player_df.columns:
                    total_minutes = player_df['player_season_minutes'].dropna().iloc[0] if not player_df['player_season_minutes'].dropna().empty else 0
                else:
                    total_minutes = 0
                    
                player_df = player_df[player_df['x'] >= 60]
                st.write(f"{selected_display_name}: Shot Map Analysis")
                
                from visual import plot_player_dashboard
                
                fig = plot_player_dashboard(
                    player_df=player_df,
                    player_agg=player_agg,
                    selected_player=selected_player,
                    selected_display_name=selected_display_name,
                    selected_team=selected_team,
                    selected_comp_name=selected_comp_name,
                    selected_season_name=selected_season_name,
                    position_filter=position_filter,
                    min_minutes=min_minutes,
                    max_minutes=max_minutes,
                    total_minutes=total_minutes,
                )
                st.pyplot(fig)

else:
    st.warning("No competitions available.")
