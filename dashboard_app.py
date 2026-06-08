"""
Tennis Analysis Dashboard

Run with:
    streamlit run dashboard_app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from court_viz import build_serve_figure

st.set_page_config(page_title="Serves Dashboard", page_icon="🎾", layout="wide")
st.title("Serves Dashboard")

DEFAULT_DIR = Path(__file__).parent / "output_videos"

with st.sidebar:

    def _load(default_name):
        default = DEFAULT_DIR / default_name
        if default.exists():
            return pd.read_csv(default)
        f = st.file_uploader(default_name, type="csv", key=default_name)
        return pd.read_csv(f) if f else None

    df_court  = _load("court_keypoints.csv")
    df_ball   = _load("ball_coords.csv")
    df_events = _load("events.csv")

    st.divider()

    is_multi = df_court is not None and "point" in df_court.columns
    if is_multi:
        all_points = sorted(df_court["point"].unique())
        selected_points = st.multiselect(
            "Points", all_points, default=all_points,
            format_func=lambda p: f"Point {p}",
        )
    else:
        selected_points = None

    selected_players = st.multiselect("Player", ["player 1", "player 2"],
                                      default=["player 1", "player 2"],
                                      format_func=lambda p: p.title(),
                                      key="filter_players")
    selected_sides   = st.multiselect("Court side", ["Deuce", "Ad"],
                                      default=["Deuce", "Ad"],
                                      key="filter_sides")

if df_court is None:
    st.info("👈 No CSV files found in output_videos/. Upload them in the sidebar.")
    st.stop()

fig = build_serve_figure(df_court, df_ball, df_events,
                         selected_players=selected_players,
                         selected_sides=selected_sides,
                         selected_points=selected_points)
st.plotly_chart(fig, use_container_width=True)

with st.expander("📋 Raw keypoints"):
    st.dataframe(df_court, use_container_width=True)
if df_events is not None:
    with st.expander("📋 Raw events"):
        st.dataframe(df_events, use_container_width=True)
