"""
Shared tennis court visualisation logic used by both app.py and dashboard_app.py.
"""

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LINE_COLORS = {
    "baseline":         "#FFFFFF",
    "doubles sideline": "#FFFFFF",
    "singles sideline": "#FFFFFF",
    "service line":     "#FFFFFF",
    "centre line":      "#FFFFFF",
    "net":              "#AADDFF",
}

SHOT_COLORS  = {"SERVE": "#FFD700", "bounce": "#FF8800"}
SHOT_SYMBOLS = {"SERVE": "star-triangle-up", "bounce": "x"}

# Real-world court coordinates (feet). Origin = back-left corner, width=36, length=78.
COURT_RW = np.array([
    [ 0,  0], [36,  0],    # 0,1  back baseline corners
    [ 0, 78], [36, 78],    # 2,3  front baseline corners
    [ 0,  0], [ 0, 78],    # 4,5  left alley
    [36,  0], [36, 78],    # 6,7  right alley
    [4.5, 0], [31.5, 0],   # 8,9  back service line ends
    [4.5,78], [31.5,78],   # 10,11 front service line ends
    [18,  0], [18, 78],    # 12,13 centre line ends
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def _build_homographies(kp_df: pd.DataFrame):
    pts = np.array(
        [[r["x"], r["y"]] for _, r in kp_df.sort_values("keypoint_id").iterrows()],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(COURT_RW, pts)
    return H, np.linalg.inv(H)


def _rw_to_img(x_rw, y_rw, H):
    out = cv2.perspectiveTransform(np.array([[[x_rw, y_rw]]], dtype=np.float32), H)
    return float(out[0][0][0]), float(out[0][0][1])


def _img_to_rw(ix, iy, H_inv):
    out = cv2.perspectiveTransform(np.array([[[ix, iy]]], dtype=np.float32), H_inv)
    return float(out[0][0][0]), float(out[0][0][1])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player_val(raw) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    return str(raw).strip()


def _court_side(x_rw: float, y_rw: float) -> str:
    """
    Return 'Deuce' or 'Ad' based on server position in real-world coordinates.

    Near side (y_rw > 39, player 2):  left of centre (x<18) → Ad,  right → Deuce
    Far  side (y_rw < 39, player 1):  left of centre (x<18) → Deuce, right → Ad
    """
    near = y_rw > 39.0
    left = x_rw < 18.0
    if near:
        return "Ad" if left else "Deuce"
    else:
        return "Deuce" if left else "Ad"

# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def build_serve_figure(
    df_court: pd.DataFrame,
    df_ball: pd.DataFrame | None,
    df_events: pd.DataFrame | None,
    selected_players: list | None = None,   # e.g. ["player 1", "player 2"]
    selected_sides: list | None = None,     # e.g. ["Deuce", "Ad"]
    selected_points: list | None = None,
) -> go.Figure:
    """
    Build the near-half tennis court figure with serve/bounce markers.
    None for any filter list means "show all".
    """
    is_multi = "point" in df_court.columns

    # Reference court keypoints
    if is_multi:
        all_pts = sorted(df_court["point"].unique())
        pts = selected_points if selected_points else all_pts
        ref_pid = pts[0] if pts else all_pts[0]
        court_kp_df = df_court[df_court["point"] == ref_pid]
    else:
        court_kp_df = df_court
        pts = None

    H_ref, H_ref_inv = _build_homographies(court_kp_df)

    # Per-point H_inv for accurate cross-point coordinate conversion
    point_H_inv: dict = {}
    if is_multi:
        for pid, grp in df_court.groupby("point"):
            try:
                _, h_inv = _build_homographies(grp)
                point_H_inv[pid] = h_inv
            except Exception:
                pass

    # Near-half boundary points
    net_l   = _rw_to_img( 0.0, 39.0, H_ref)
    net_r   = _rw_to_img(36.0, 39.0, H_ref)
    net_ls  = _rw_to_img( 4.5, 39.0, H_ref)
    net_rs  = _rw_to_img(31.5, 39.0, H_ref)
    net_c   = _rw_to_img(18.0, 39.0, H_ref)

    kp = {int(r["keypoint_id"]): (float(r["x"]), float(r["y"]))
          for _, r in court_kp_df.iterrows()}

    # ── Prepare events ──────────────────────────────────────────────────────
    if df_events is not None and len(df_events) > 0:
        ev = df_events[df_events["event"].isin(["SERVE", "bounce"])].copy()

        if is_multi and pts is not None:
            ev = ev[ev["point"].isin(pts)]

        ev["_player"]     = ev["player"].apply(_player_val)
        ev["_court_side"] = ""   # will be filled below

        # Walk events in order: for each SERVE compute player + court_side,
        # then propagate both to the immediately following bounce.
        src_sorted = df_events.sort_values("frame").reset_index(drop=True)
        last_player: dict = {}
        last_side:   dict = {}

        for _, row in src_sorted.iterrows():
            pid = row["point"] if (is_multi and "point" in row.index
                                   and not pd.isna(row.get("point"))) else "_"
            if row["event"] == "SERVE":
                player = _player_val(row["player"])
                last_player[pid] = player

                # Compute real-world position of serve contact to determine side
                if df_ball is not None:
                    frame = int(row["frame"])
                    if is_multi and "point" in df_ball.columns:
                        brow = df_ball[(df_ball["point"] == row["point"]) &
                                       (df_ball["frame"] == frame)]
                    else:
                        brow = df_ball[df_ball["frame"] == frame]

                    if not brow.empty:
                        bx = float(brow["ball_x"].iloc[0])
                        by = float(brow["ball_y"].iloc[0])
                        if not (np.isnan(bx) or np.isnan(by)):
                            src_pid = row["point"] if (is_multi and "point" in row.index
                                                        and not pd.isna(row.get("point"))) else None
                            h_inv = point_H_inv.get(src_pid, H_ref_inv)
                            x_rw, y_rw = _img_to_rw(bx, by, h_inv)
                            side = _court_side(x_rw, y_rw)
                            last_side[pid] = side

                            # Write into ev
                            mask = ev["frame"] == frame
                            if is_multi and "point" in ev.columns:
                                mask &= ev["point"] == row["point"]
                            ev.loc[mask, "_court_side"] = side

            elif row["event"] == "bounce":
                mask = ev["frame"] == row["frame"]
                if is_multi and "point" in ev.columns:
                    mask &= ev["point"] == row["point"]
                if pid in last_player:
                    ev.loc[mask, "_player"] = last_player[pid]
                if pid in last_side:
                    ev.loc[mask, "_court_side"] = last_side[pid]

        # Apply filters
        if selected_players:
            ev = ev[ev["_player"].isin(selected_players)]
        if selected_sides:
            ev = ev[ev["_court_side"].isin(selected_sides)]

        # Join ball positions
        if df_ball is not None and len(df_ball) > 0:
            if is_multi and "point" in ev.columns and "point" in df_ball.columns:
                ev = ev.merge(df_ball[["point", "frame", "ball_x", "ball_y"]],
                              on=["point", "frame"], how="left")
            else:
                ev = ev.merge(df_ball[["frame", "ball_x", "ball_y"]],
                              on="frame", how="left")
            ev = ev.dropna(subset=["ball_x", "ball_y"])
    else:
        ev = pd.DataFrame()

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = go.Figure()

    near_poly_x = [net_l[0], net_r[0], kp[3][0], kp[2][0], net_l[0]]
    near_poly_y = [net_l[1], net_r[1], kp[3][1], kp[2][1], net_l[1]]
    all_x = near_poly_x + [kp[2][0], kp[3][0]]
    all_y = near_poly_y + [kp[2][1], kp[3][1]]
    pad_x = (max(all_x) - min(all_x)) * 0.12
    pad_y = (max(all_y) - min(all_y)) * 0.15
    x_min, x_max = min(all_x) - pad_x, max(all_x) + pad_x
    y_min, y_max = min(all_y) - pad_y, max(all_y) + pad_y

    fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=x_max, y1=y_max,
                  fillcolor="#1a3a28", line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=near_poly_x, y=near_poly_y,
        fill="toself", fillcolor="#2D6A4F",
        line=dict(width=0), mode="lines",
        hoverinfo="skip", showlegend=False,
    ))

    for pa, pb, group, lw in [
        (kp[2],   kp[3],   "baseline",         3),
        (net_l,   kp[2],   "doubles sideline",  3),
        (net_r,   kp[3],   "doubles sideline",  3),
        (net_ls,  kp[5],   "singles sideline",  2),
        (net_rs,  kp[7],   "singles sideline",  2),
        (kp[10],  kp[11],  "service line",      2),
        (net_c,   kp[13],  "centre line",       2),
        (net_l,   net_r,   "net",               4),
    ]:
        fig.add_trace(go.Scatter(
            x=[pa[0], pb[0]], y=[pa[1], pb[1]],
            mode="lines",
            line=dict(color=LINE_COLORS.get(group, "#FFF"), width=lw,
                      dash="dash" if group == "net" else "solid"),
            hoverinfo="skip", showlegend=False,
        ))

    if not ev.empty:
        for event_type, gdf in ev.groupby("event"):
            player_col = gdf["_player"]     if "_player"     in gdf.columns else pd.Series([""] * len(gdf))
            side_col   = gdf["_court_side"] if "_court_side" in gdf.columns else pd.Series([""] * len(gdf))
            point_col  = gdf["point"].astype(str) if "point" in gdf.columns else pd.Series([""] * len(gdf))

            xs, ys, labels = [], [], []
            for (_, row), player, side, point in zip(
                    gdf.iterrows(), player_col, side_col, point_col):
                ix, iy = float(row["ball_x"]), float(row["ball_y"])

                src_pid = row["point"] if ("point" in row.index
                                           and not pd.isna(row.get("point"))) else None
                h_inv_src = point_H_inv.get(src_pid, H_ref_inv)

                x_rw, y_rw = _img_to_rw(ix, iy, h_inv_src)
                if y_rw < 39.0:
                    x_rw, y_rw = 36.0 - x_rw, 78.0 - y_rw
                ix, iy = _rw_to_img(x_rw, y_rw, H_ref)

                xs.append(ix)
                ys.append(iy)
                labels.append(
                    f"Point {point}<br>Frame {int(row['frame'])}<br>{event_type}"
                    + (f"<br>{player}" if player else "")
                    + (f"<br>{side}" if side else "")
                )

            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(
                    color=SHOT_COLORS.get(event_type, "#FF00FF"),
                    size=14,
                    symbol=SHOT_SYMBOLS.get(event_type, "circle"),
                    line=dict(color="white", width=1.5),
                ),
                name=event_type.title(),
                hovertemplate="%{text}<extra></extra>",
                text=labels,
            ))

    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a3a28",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="y", scaleratio=1, range=[x_min, x_max]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[y_max, y_min]),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        autosize=True,
    )
    return fig
