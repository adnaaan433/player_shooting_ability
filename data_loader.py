import streamlit as st
import pandas as pd
from mplsoccer import Sbapi
import concurrent.futures

@st.cache_resource
def get_api():
    try:
        username = st.secrets["statsbomb"]["username"]
        password = st.secrets["statsbomb"]["password"]
        return Sbapi(username=username, password=password, dataframe=True)
    except Exception:
        return Sbapi(dataframe=True)

@st.cache_data
def load_competitions():
    api = get_api()
    df_comps = api.competition()
    return df_comps

@st.cache_data
def load_matches(competition_id, season_id):
    api = get_api()
    matches = api.match(competition_id, season_id)
    return matches

@st.cache_data(show_spinner="Downloading Player Stats from API...")
def load_player_season_stats(competition_id, season_id):
    import requests
    from requests.auth import HTTPBasicAuth
    
    url = f"https://data.statsbombservices.com/api/v4/competitions/{competition_id}/seasons/{season_id}/player-stats"
    try:
        username = st.secrets["statsbomb"]["username"]
        password = st.secrets["statsbomb"]["password"]
        resp = requests.get(url, auth=HTTPBasicAuth(username, password))
        if resp.status_code == 200:
            df = pd.DataFrame(resp.json())
            df['player_known_name'] = df['player_known_name'].fillna(df['player_name'])
            
            # Ensure columns exist before filtering, in case endpoint format shifts
            keep_cols = ['player_name', 'team_name', 'player_known_name', 'player_season_minutes', 'primary_position']
            existing_cols = [c for c in keep_cols if c in df.columns]
            
            return df[existing_cols]
        else:
            st.error(f"Failed to load player stats: HTTP {resp.status_code}")
    except Exception as e:
        st.error(f"API Error fetching player stats: {e}")
        
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_filtered_comp_event(match_id):
    api = get_api()
    try:
        res = api.event(match_id)
        if res is not None:
            df = res[0]
            if 'type_name' in df.columns:
                df = df[df['type_name'] == 'Shot']
            else:
                return None
            if 'sub_type_name' in df.columns:
                df = df[df['sub_type_name'] != 'Penalty']
            target_cols = ['type_name', 'sub_type_name', 'outcome_name', 'player_name', 'position_name', 'team_name', 'play_pattern_name', 'body_part_name',
                            'x', 'y', 'z', 'end_x', 'end_y', 'end_z', 'shot_statsbomb_xg', 'shot_statsbomb_xg2',
                            'under_pressure', 'shot_first_time', 'shot_follows_dribble', 'shot_one_on_one']
            existing_cols = [c for c in target_cols if c in df.columns]
            return df[existing_cols]
    except Exception:
        pass
    return None

def load_competition_events_from_api(competition_id, season_id, progress_bar=None, status_text=None):
    df_matches = load_matches(competition_id, season_id)
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()
        
    if 'match_status' in df_matches.columns:
        match_ids = df_matches[df_matches['match_status'] == 'available']['match_id'].tolist()
    else:
        match_ids = df_matches['match_id'].tolist()
        
    events_list = []
    total = len(match_ids)
    
    if total == 0:
        return pd.DataFrame()
        
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(fetch_filtered_comp_event, mid) for mid in match_ids]
        for future in concurrent.futures.as_completed(futures):
            df_ep = future.result()
            if df_ep is not None and not df_ep.empty:
                events_list.append(df_ep)
                
            completed += 1
            if progress_bar is not None:
                progress_bar.progress(completed / total)
            if status_text is not None:
                status_text.text(f"Downloading events: {completed}/{total} matches completed...")
                
    if events_list:
        return pd.concat(events_list, ignore_index=True)
    return pd.DataFrame()
