import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc
from matplotlib.font_manager import FontProperties
from mplsoccer import VerticalPitch, add_image
from scipy.stats import percentileofscore
from urllib.request import urlopen
from PIL import Image

_DIR = os.path.dirname(os.path.abspath(__file__))
bold_font = FontProperties(fname=os.path.join(_DIR, 'MontserratAlternates-Bold.ttf'))
reg_font = FontProperties(fname=os.path.join(_DIR, 'NotoSans-Regular.ttf'))
con_font = FontProperties(fname=os.path.join(_DIR, 'NotoSans_Condensed-Regular.ttf'))

def plot_player_dashboard(player_df, player_agg, selected_player, selected_display_name,
                          selected_team, selected_comp_name, selected_season_name,
                          position_filter, min_minutes, max_minutes, total_minutes):
    """
    Create the full player shooting dashboard figure.
    
    Parameters:
        player_df: DataFrame of the selected player's shot events (already filtered to x >= 60)
        player_agg: DataFrame of aggregated player stats (all players, for percentile calculations)
        selected_player: str, the player_name key
        selected_display_name: str, display name for the title
        selected_team: str
        selected_comp_name: str
        selected_season_name: str
        position_filter: str
        min_minutes: int
        total_minutes: int/float
        
    Returns:
        matplotlib Figure
    """
    
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor('#f5f5f5')
    
    gs = gridspec.GridSpec(4, 2, width_ratios=[10, 7.5], height_ratios=[22.5, 22.5, 70, 12],
                           wspace=0, hspace=0.1, bottom=0.03)
    
    # ── GoalPost Stats (heatmap) ──
    goalpost_stats = fig.add_subplot(gs[0, 0])
    goalpost_stats.set_facecolor('#f5f5f5')
    goalpost_stats.set_title('GoalPost View', fontproperties=bold_font, fontsize=15)
    for spine in goalpost_stats.spines.values():
        spine.set_visible(False)
    goalpost_stats.tick_params(bottom=False, left=False, right=False, top=False,
                               labelbottom=False, labelleft=False)
    
    # ── GoalPost Viz (shot scatter) ──
    goalpost_viz = fig.add_subplot(gs[1, 0])
    goalpost_viz.set_facecolor('#f5f5f5')
    for spine in goalpost_viz.spines.values():
        spine.set_visible(False)
    goalpost_viz.tick_params(bottom=False, left=False, right=False, top=False,
                              labelbottom=False, labelleft=False)
    
    # Draw goalpost
    goalpost_viz.plot([36, 36, 44, 44], [0, 2.67, 2.67, 0], color='black', linewidth=5, zorder=3)
    goalpost_viz.plot([32, 48], [0, 0], color='black', linewidth=3, zorder=3)
    
    # Draw net inside goalpost_viz
    for x_line in np.linspace(36, 44, 11)[1:-1]:
        goalpost_viz.plot([x_line, x_line], [0, 2.67], color='gray', linestyle='--',
                          linewidth=0.8, alpha=0.5, zorder=1)
    for y_line in np.linspace(0, 2.67, 7)[1:-1]:
        goalpost_viz.plot([36, 44], [y_line, y_line], color='gray', linestyle='--',
                          linewidth=0.8, alpha=0.5, zorder=1)
    
    goalpost_viz.set_xlim(32, 48)
    goalpost_viz.set_ylim(-0.75, 3.5)
    
    # ── Pitch View ──
    pitch_ax = fig.add_subplot(gs[2, 0])
    pitch_ax.set_title('Pitch View', fontproperties=bold_font, fontsize=15)
    pitch = VerticalPitch(pitch_type='statsbomb', half=True, pitch_color='#f5f5f5')
    pitch.draw(ax=pitch_ax)
    pitch_ax.set_aspect('auto')  # Stretch pitch to match goalpost axes width
    
    # ── Prepare shot data ──
    total_shots = len(player_df)
    
    def get_label(name, count):
        if total_shots == 0:
            return f"{name}: 0 (0%)"
        pct = (count / total_shots) * 100
        return f"{name}: {count} ({pct:.0f}%)"
    
    player_df = player_df.copy()
    if 'shot_statsbomb_xg2' in player_df.columns:
        player_df['plot_xg'] = player_df['shot_statsbomb_xg2'].fillna(player_df['shot_statsbomb_xg'])
    else:
        player_df['plot_xg'] = player_df['shot_statsbomb_xg']
    
    if 'z' not in player_df.columns:
        player_df['z'] = 0
    else:
        player_df['z'] = player_df['z'].fillna(0)
    
    dx = player_df['end_x'] - player_df['x']
    dx = dx.replace(0, 0.001)
    t = (120 - player_df['x']) / dx
    player_df['est_y'] = player_df['y'] + t * (player_df['end_y'] - player_df['y'])
    player_df['est_z'] = player_df['z'] + t * (player_df['end_z'] - player_df['z'])
    
    # ── GoalPost Stats Heatmap ──
    goalpost_stats.plot([36, 36, 44, 44], [0, 2.67, 2.67, 0], color='black', linewidth=5, zorder=3)
    goalpost_stats.plot([32, 48], [0, 0], color='black', linewidth=3, zorder=3)
    goalpost_stats.set_xlim(32, 48)
    goalpost_stats.set_ylim(-0.25, 3.75)
    
    valid_shots = player_df[player_df['outcome_name'] != 'Blocked']
    x_bins = np.linspace(36, 44, 6)
    y_bins = np.linspace(0, 2.67, 4)
    
    H, xedges, yedges = np.histogram2d(valid_shots['est_y'], valid_shots['est_z'],
                                        bins=[x_bins, y_bins])
    total_in_grid = np.sum(H)
    
    X, Y = np.meshgrid(xedges, yedges)
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#f5f5f5", "orange"])
    goalpost_stats.pcolormesh(X, Y, H.T, cmap=custom_cmap, alpha=0.9,
                              edgecolors='white', linewidth=1, zorder=1)
    
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            count = int(H[i, j])
            if total_in_grid > 0:
                pct = (count / total_in_grid) * 100
            else:
                pct = 0
            cx = (x_bins[i] + x_bins[i + 1]) / 2
            cy = (y_bins[j] + y_bins[j + 1]) / 2
            if count > 0:
                goalpost_stats.text(cx, cy, f"{count}\n({pct:.0f}%)", ha='center',
                                    va='center', color='black', zorder=2,
                                    fontproperties=reg_font, fontsize=11)
    
    # ── Plot Shots on Pitch & Goalpost ──
    goals = player_df[player_df['outcome_name'] == 'Goal']
    if not goals.empty:
        pitch.scatter(goals['x'], goals['y'], s=goals['plot_xg'] * 250 + 50,
                      marker='o', color='green', edgecolors='white', ax=pitch_ax,
                      zorder=6, label=get_label("Goal", len(goals)))
        goalpost_viz.scatter(goals['est_y'], goals['est_z'], s=goals['plot_xg'] * 250 + 50,
                             marker='o', color='green', edgecolors='white', zorder=6)
    else:
        pitch.scatter([], [], s=100, marker='o', color='green', edgecolors='white',
                      ax=pitch_ax, label=get_label("Goal", 0))
    
    saved = player_df[player_df['outcome_name'].isin(['Saved', 'Saved to Post', 'Saved To Post'])]
    if not saved.empty:
        pitch.scatter(saved['x'], saved['y'], s=saved['plot_xg'] * 250 + 50,
                      marker='o', hatch='//////////', facecolors='none', edgecolors='orange',
                      ax=pitch_ax, zorder=5, label=get_label("Saved", len(saved)))
        goalpost_viz.scatter(saved['est_y'], saved['est_z'], s=saved['plot_xg'] * 250 + 50,
                             marker='o', hatch='//////////', facecolors='none',
                             edgecolors='orange', zorder=5)
    else:
        pitch.scatter([], [], s=100, marker='o', hatch='//////////', facecolors='none',
                      edgecolors='orange', ax=pitch_ax, label=get_label("Saved", 0))
    
    off_t = player_df[player_df['outcome_name'].isin(['Off T', 'Saved Off T', 'Post', 'Wayward'])]
    if not off_t.empty:
        pitch.scatter(off_t['x'], off_t['y'], s=off_t['plot_xg'] * 250 + 50,
                      marker='o', facecolors='none', edgecolors='orange', ax=pitch_ax,
                      zorder=4, label=get_label("Off Target", len(off_t)))
        goalpost_viz.scatter(off_t['est_y'], off_t['est_z'], s=off_t['plot_xg'] * 250 + 50,
                             marker='o', facecolors='none', edgecolors='orange', zorder=4)
    else:
        pitch.scatter([], [], s=100, marker='o', facecolors='none', edgecolors='orange',
                      ax=pitch_ax, label=get_label("Off Target", 0))
    
    blocked = player_df[player_df['outcome_name'] == 'Blocked']
    if not blocked.empty:
        pitch.scatter(blocked['x'], blocked['y'], s=blocked['plot_xg'] * 250 + 50,
                      marker='x', color='orange', ax=pitch_ax, zorder=3,
                      label=get_label("Blocked", len(blocked)))
    else:
        pitch.scatter([], [], s=100, marker='x', color='orange', ax=pitch_ax,
                      label=get_label("Blocked", 0))
    
    # ── Distance Arcs ──
    arc1 = Arc((40, 120), width=20, height=20, theta1=175, theta2=365,
               color='skyblue', linestyle='--', lw=2.5, zorder=2)
    arc2 = Arc((40, 120), width=36, height=36, theta1=175, theta2=365,
               color='skyblue', linestyle='--', lw=2.5, zorder=2)
    arc3 = Arc((40, 120), width=56, height=56, theta1=175, theta2=365,
               color='skyblue', linestyle='--', lw=2.5, zorder=2)
    
    pitch_ax.add_patch(arc1)
    pitch_ax.add_patch(arc2)
    pitch_ax.add_patch(arc3)
    
    bbox_props = dict(boxstyle="round,pad=0.1", fc="#f5f5f5", ec="none", alpha=0.9)
    pitch_ax.text(50, 122.5, "10m", color='steelblue', fontsize=10, fontproperties=bold_font,
                  ha='center', va='top', bbox=bbox_props, zorder=10)
    pitch_ax.text(58, 122.5, "18m", color='steelblue', fontsize=10, fontproperties=bold_font,
                  ha='center', va='top', bbox=bbox_props, zorder=10)
    pitch_ax.text(68, 122.5, "28m", color='steelblue', fontsize=10, fontproperties=bold_font,
                  ha='center', va='top', bbox=bbox_props, zorder=10)
    
    # ── Legend ──
    # pitch_ax.text(40, 62.5, "@adnaaan433", alpha=0.25, fontsize=20, ha='center', va='center', fontweight='bold')
    pitch_ax.text(40, 62.5, "*Circle Size = xG*", fontsize=12, ha='center', va='center',
                  fontproperties=reg_font)
    legend_font = reg_font.copy()
    legend_font.set_size(12)
    legend = pitch_ax.legend(loc='lower right', facecolor='#f5f5f5', edgecolor='none',
                             labelcolor='black', prop=legend_font)
    handles = getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', []))
    for handle in handles:
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([75])
    
    # ── Shot Zone Stats ──
    if total_shots > 0:
        dist = np.sqrt((player_df['x'] - 120) ** 2 + (player_df['y'] - 40) ** 2)
        zone_10 = sum(dist < 10)
        zone_18 = sum((dist >= 10) & (dist < 18))
        zone_28 = sum((dist >= 18) & (dist < 28))
        zone_out = sum(dist >= 28)
        
        pct_10 = (zone_10 / total_shots) * 100
        pct_18 = (zone_18 / total_shots) * 100
        pct_28 = (zone_28 / total_shots) * 100
        pct_out = (zone_out / total_shots) * 100
    else:
        zone_10 = zone_18 = zone_28 = zone_out = 0
        pct_10 = pct_18 = pct_28 = pct_out = 0
    
    stats_text = (
        "Shot Zone:\n"
        f"<10m radius: {zone_10} ({pct_10:.0f}%)\n"
        f"<18m radius: {zone_18} ({pct_18:.0f}%)\n"
        f"<28m radius: {zone_28} ({pct_28:.0f}%)\n"
        f">28m radius: {zone_out} ({pct_out:.0f}%)"
    )
    pitch_ax.text(0.02, 0.02, stats_text, transform=pitch_ax.transAxes,
                  color='black', fontsize=12, va='bottom', ha='left',
                  bbox=dict(facecolor='#f5f5f5', edgecolor='none', alpha=0.8),
                  fontproperties=reg_font)
    
    # ── Shooting Stats Panel (right side) ──
    stats_ax = fig.add_subplot(gs[0:3, 1])
    stats_ax.set_facecolor('#f5f5f5')
    
    if selected_player in player_agg.index:
        stats_ax.set_title("Shooting Stats (per90)", color='black',
                           fontproperties=bold_font, fontsize=15, pad=0)
        for spine in stats_ax.spines.values():
            spine.set_visible(False)
        stats_ax.tick_params(bottom=False, left=False, right=False, top=False,
                             labelbottom=False, labelleft=False)
        
        metrics_to_plot = [
            'Total Shots per90',
            'Total Goals per90',
            'Total xG per90',
            'Total xGOT per90',
            'xG Overperformance',
            'Avg. Shot Distance',
            'Total Shot Accuracy',
            'Shot accuracy under pressure',
            'Low Chance Shots per90',
            'Low Chance Conversion%',
            'Half Chance Shots per90',
            'Half Chance Conversion%',
            'Big Chance Shots per90',
            'Big Chance Conversion%',
        ]
        
        p_vals = []
        p_pcts = []
        for m in metrics_to_plot:
            val = player_agg.loc[selected_player, m]
            pct = percentileofscore(player_agg[m].dropna(), val)
            
            # Invert percentile for Avg. Shot Distance (lower distance = higher percentile)
            if m == 'Avg. Shot Distance':
                pct = 100 - pct
            
            p_vals.append(val)
            p_pcts.append(pct)
        
        y_pos = np.arange(len(metrics_to_plot))[::-1]
        
        # Background bars
        stats_ax.barh(y_pos, [100] * len(metrics_to_plot), color='#e0e0e0', height=0.15, zorder=1)
        
        # Foreground bars (colored by percentile)
        cmap = LinearSegmentedColormap.from_list("custom_stats", ['red', 'orange', 'green'])
        colors = [cmap(p / 100.0) for p in p_pcts]
        stats_ax.barh(y_pos, p_pcts, color=colors, height=0.15, zorder=2)
        
        # Scatter at end of bar with percentile value
        for y, p, c in zip(y_pos, p_pcts, colors):
            stats_ax.scatter(p, y, color=c, s=400, zorder=3, edgecolors='white', linewidth=1.5)
            stats_ax.text(p, y, f"{p:.0f}", color='white', fontsize=9, ha='center',
                          va='center', fontproperties=bold_font, zorder=4)
        
        # Display name mapping
        display_name_map = {
            'Total Shots': 'npShots',
            'Total Goals': 'npGoals',
            'Total xG': 'npxG',
            'Total xGOT': 'npxGOT',
            'xG Overperformance': 'npGoals - npxG',
            'Avg. Shot Distance': 'Avg. npShot Distance',
            'Total Shot Accuracy': 'npShot Accuracy',
            'Shot accuracy under pressure': 'npShot Accuracy (Under Pressure)',
            'Low Chance Shots': 'Low Chance Shots',
            'Low Chance Conversion%': 'Low Chance Conversion%',
            'Half Chance Shots': 'Half Chance Shots',
            'Half Chance Conversion%': 'Half Chance Conversion%',
            'Big Chance Shots': 'Big Chance Shots',
            'Big Chance Conversion%': 'Big Chance Conversion%',
        }
        
        # Text labels
        for y, m, val, pct in zip(y_pos, metrics_to_plot, p_vals, p_pcts):
            if 'per90' in m:
                base_m = m.replace(' per90', '')
                base_val = player_agg.loc[selected_player, base_m]
                display_base = display_name_map.get(base_m, base_m)
                if base_m in ['Total Shots', 'Total Goals', 'Low Chance Shots',
                              'Half Chance Shots', 'Big Chance Shots']:
                    base_val_str = f"{int(base_val)}"
                    p90_val_str = f"{val:.2f}"
                else:
                    base_val_str = f"{base_val:.2f}"
                    p90_val_str = f"{val:.2f}"
                display_text = f"{display_base} (per90): {base_val_str} ({p90_val_str})"
            else:
                display_name = display_name_map.get(m, m)
                if m == 'xG Overperformance':
                    display_text = f"{display_name}: {val:.2f}"
                elif m == 'Avg. Shot Distance':
                    display_text = f"{display_name}: {val:.1f}m"
                elif 'Conversion%' in m:
                    display_text = f"{display_name}: {val * 100:.1f}%"
                else:
                    display_text = f"{display_name}: {val * 100:.1f}%"
            
            stats_ax.text(0, y + 0.25, display_text, ha='left', va='bottom',
                          fontsize=11, color='black', zorder=3, fontproperties=reg_font)
        
        stats_ax.set_xlim(-5, 105)
        stats_ax.set_ylim(-0.8, len(metrics_to_plot))
    
    else:
        stats_ax.text(0.5, 0.5, "Player does not meet\nminimum minutes filter",
                      ha='center', va='center', fontsize=15, color='gray',
                      fontproperties=reg_font)
        for spine in stats_ax.spines.values():
            spine.set_visible(False)
        stats_ax.tick_params(bottom=False, left=False, right=False, top=False,
                             labelbottom=False, labelleft=False)
    
    # ── Bottom Axes: Body Part & Situation Stats ──
    bottom_ax = fig.add_subplot(gs[3, :])
    bottom_ax.set_facecolor('#f5f5f5')
    for spine in bottom_ax.spines.values():
        spine.set_visible(False)
    bottom_ax.tick_params(bottom=False, left=False, right=False, top=False,
                          labelbottom=False, labelleft=False)
    
    if selected_player in player_agg.index:
        body_part_stats = [
            ('Right Foot Shots', 'Right Foot Shots Accuracy'),
            ('Left Foot Shots', 'Left Foot Shots Accuracy'),
            ('Headed Shots', 'Headed Shots Accuracy'),
        ]
        situation_stats = [
            ('First Touch Shots', 'First Touch Conversion%'),
            ('One on One Shots', 'One on One Conversion%'),
            ('Dribbled Shots', 'Dribbled Conversion%'),
        ]
        
        # Title line
        bottom_ax.text(0.25, 0.9, "Body Part (Shot Accuracy%)", ha='center', va='center',
                       fontsize=15, fontproperties=bold_font, color='black', transform=bottom_ax.transAxes)
        bottom_ax.text(0.75, 0.9, "Situation (Conversion%)", ha='center', va='center',
                       fontsize=15, fontproperties=bold_font, color='black', transform=bottom_ax.transAxes)
        
        # Separator line
        bottom_ax.plot([0.5, 0.5], [0.4, 0.95], color='#cccccc', linewidth=1,
                       transform=bottom_ax.transAxes, clip_on=False)
        
        # Body Part stats line
        bp_parts = []
        for vol_col, acc_col in body_part_stats:
            vol = int(player_agg.loc[selected_player, vol_col]) if vol_col in player_agg.columns else 0
            acc = player_agg.loc[selected_player, acc_col] * 100 if acc_col in player_agg.columns else 0.0
            label = vol_col.replace(' Shots', '')
            bp_parts.append(f"{label}: {vol} ({acc:.1f}%)")
        bottom_ax.text(0.25, 0.55, "   |   ".join(bp_parts), ha='center', va='center',
                       fontsize=12, color='black', transform=bottom_ax.transAxes,
                       fontproperties=reg_font)
        
        # Situation stats line
        sit_parts = []
        for vol_col, conv_col in situation_stats:
            vol = int(player_agg.loc[selected_player, vol_col]) if vol_col in player_agg.columns else 0
            conv = player_agg.loc[selected_player, conv_col] * 100 if conv_col in player_agg.columns else 0.0
            label = vol_col.replace(' Shots', '')
            sit_parts.append(f"{label}: {vol} ({conv:.1f}%)")
        bottom_ax.text(0.75, 0.55, "   |   ".join(sit_parts), ha='center', va='center',
                       fontsize=12, color='black', transform=bottom_ax.transAxes,
                       fontproperties=reg_font)
    
    # ── Logo ──
    _teams_csv = os.path.join(_DIR, 'teams_name_and_id_Statsbomb_Names.csv')
    _teams_df = pd.read_csv(_teams_csv, index_col=0)
    _match = _teams_df[_teams_df['teamName'] == selected_team]
    ftmb_tid = int(_match['teamId'].iloc[0]) if not _match.empty else None
    if ftmb_tid:
        try:
            himage = urlopen(f"https://images.fotmob.com/image_resources/logo/teamlogo/{ftmb_tid}.png")
            himage = Image.open(himage)
            ax_himage = add_image(himage, fig, left=0.115, bottom=0.925, width=0.12, height=0.12)
        except Exception:
            pass

    # ── Title Text ──
    fig.text(0.23, 1, f"{selected_display_name}",
             fontsize=30, fontproperties=bold_font, color='black', ha='left', va='bottom')
    fig.text(0.23, 0.97,
             f"for {selected_team}, in {selected_comp_name} {selected_season_name} season | "
             f"Minutes Played: {int(total_minutes)} | Data: StatsBomb | Made by: @adnaaan433",
             ha='left', va='bottom', fontsize=13, fontproperties=con_font)
    minutes_label = f"{min_minutes}+" if max_minutes >= 3500 else f"{min_minutes}-{max_minutes}"
    if isinstance(position_filter, list):
        if not position_filter or set(position_filter) == {'CF', 'Winger/AM', 'Mid', 'FB', 'CB'}:
            pos_desc = "All"
        else:
            pos_desc = f"{'/'.join(position_filter)}s"
    else:
        pos_desc = f"{position_filter}s" if position_filter != 'All' else "All"
        
    fig.text(0.23, 0.94,
             f"Non-Penalty Shots Only | Percentiles among {selected_comp_name} {pos_desc} "
             f"with {minutes_label} Minutes Played in {selected_season_name} season",
             ha='left', va='bottom', fontsize=13, fontproperties=con_font)
    
    return fig
