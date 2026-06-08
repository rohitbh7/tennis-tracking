"""
Streamlit front-end for the tennis analysis pipeline.

Run with:
    streamlit run app.py
"""

import csv
import io
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import pandas as pd

from court_viz import build_serve_figure
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Project-root helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()


def _in_project_root():
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        prev = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            yield
        finally:
            os.chdir(prev)

    return _ctx()


# ---------------------------------------------------------------------------
# CSV helpers — each accepts a point_id and appends it as the first column
# ---------------------------------------------------------------------------

_COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

_COURT_KEYPOINT_NAMES = [
    "back-left baseline corner", "back-right baseline corner",
    "front-left baseline corner", "front-right baseline corner",
    "left alley back", "left alley front",
    "right alley back", "right alley front",
    "back service line left", "back service line right",
    "front service line left", "front service line right",
    "service box centre back", "service box centre front",
]


def _rows_events(shot_frames: dict, bounce_frames: dict, point_id: int) -> list:
    rows = []
    for frame, data in shot_frames.items():
        label = data[2]
        player_num = data[3]
        rows.append([point_id, frame, label, f"player {player_num}" if player_num else ""])
    for frame in bounce_frames:
        rows.append([point_id, frame, "bounce", ""])
    rows.sort(key=lambda r: r[1])
    return rows


def _rows_ball(ball_detections: list, point_id: int) -> list:
    return [[point_id, i, x, y] for i, (x, y) in enumerate(ball_detections)]


def _rows_pose(pose_detections: list, point_id: int) -> list:
    rows = []
    for frame_idx, kp in enumerate(pose_detections):
        if kp is None or kp.xy.shape[0] < 2:
            continue
        p1, p2 = kp.xy[0], kp.xy[1]
        for ji in range(min(len(p1), len(p2))):
            name = _COCO_JOINT_NAMES[ji] if ji < len(_COCO_JOINT_NAMES) else str(ji)
            rows.append([point_id, frame_idx, ji, name,
                         p1[ji][0], p1[ji][1], p2[ji][0], p2[ji][1]])
    return rows


def _rows_court(court_keypoints, point_id: int) -> list:
    rows = []
    for i in range(14):
        x = court_keypoints[i * 2]
        y = court_keypoints[i * 2 + 1]
        name = _COURT_KEYPOINT_NAMES[i] if i < len(_COURT_KEYPOINT_NAMES) else str(i)
        rows.append([point_id, i, name, x, y])
    return rows


def _build_csvs(all_clips: list) -> tuple[str, str, str, str]:
    """
    all_clips: list of dicts with keys
        point_id, shot_frames, bounce_frames, ball_detections,
        pose_detections, court_keypoints
    Returns four CSV strings: events, ball, pose, court.
    """
    ev_rows, ba_rows, po_rows, co_rows = [], [], [], []
    for c in all_clips:
        pid = c["point_id"]
        ev_rows.extend(_rows_events(c["shot_frames"], c["bounce_frames"], pid))
        ba_rows.extend(_rows_ball(c["ball_detections"], pid))
        po_rows.extend(_rows_pose(c["pose_detections"], pid))
        co_rows.extend(_rows_court(c["court_keypoints"], pid))

    def _dump(header, rows):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(header)
        w.writerows(rows)
        return buf.getvalue()

    return (
        _dump(["point", "frame", "event", "player"],                                         ev_rows),
        _dump(["point", "frame", "ball_x", "ball_y"],                                        ba_rows),
        _dump(["point", "frame", "joint", "joint_name",
               "player1_x", "player1_y", "player2_x", "player2_y"],                          po_rows),
        _dump(["point", "keypoint_id", "court_location", "x", "y"],                          co_rows),
    )


# ---------------------------------------------------------------------------
# Single-clip pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_video_path: str,
    p1_handedness: str,
    p2_handedness: str,
    output_video_path: str,
    status,
    clip_label: str = "",
) -> dict:
    """
    Run the full pipeline for one clip.
    Returns a dict with all raw results needed for CSV building and display.
    """
    prefix = f"[{clip_label}] " if clip_label else ""

    def _step(msg: str):
        if status is not None:
            status.update(label=f"{prefix}{msg}")

    from court_line_detector.court_line_detector import CourtLineDetector
    from trackers import (
        BallTracker, BounceTracker, PlayerTracker, PoseDetector, ShotTracker2,
    )
    from utils import read_video, save_video

    _step("📽️ Reading video…")
    video_frames = read_video(input_video_path)
    if not video_frames:
        raise ValueError(f"Could not read any frames from: {input_video_path}")

    _step("🎾 Detecting court lines…")
    court_detector = CourtLineDetector("keypoints_model.pth")
    court_keypoints = court_detector.predict(video_frames[0])

    _step("🏃 Tracking players…")
    player_tracker = PlayerTracker(model_path="yolo12n.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False)
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    _step("🦾 Estimating poses…")
    pose_detector = PoseDetector()
    pose_detections = pose_detector.detect_frames(video_frames, player_detections=player_detections)

    _step("🟡 Tracking ball…")
    ball_tracker = BallTracker(model_path="model_best.pt", device="cpu")
    ball_detections = ball_tracker.detect_frames(video_frames, extrapolation=True)

    _step("🏓 Detecting shots…")
    frame_height, frame_width = video_frames[0].shape[:2]
    shot_tracker = ShotTracker2(
        wrist_proximity_px=100.0,
        min_velocity_change=12.0,
        min_post_speed=7.0,
        persist_frames=20,
        p1_handedness=p1_handedness,
        p2_handedness=p2_handedness,
    )
    shot_frames = shot_tracker.detect_shots(
        ball_detections, pose_detections,
        frame_height=frame_height, frame_width=frame_width,
    )

    _step("⚡ Detecting bounces…")
    bounce_tracker = BounceTracker()
    bounce_frames = bounce_tracker.detect_bounces(
        ball_detections, shot_frames=shot_frames, pose_detections=pose_detections,
    )

    _step("🎨 Rendering output video…")
    out_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    out_frames = [court_detector.draw_keypoints(f, court_keypoints) for f in out_frames]
    out_frames = ball_tracker.draw_bboxes(out_frames, ball_detections)
    out_frames = bounce_tracker.draw_bounces(out_frames, bounce_frames)
    out_frames = shot_tracker.draw_shot_markers(out_frames, shot_frames)
    out_frames = pose_detector.draw_poses(out_frames, pose_detections)

    for i, frame in enumerate(out_frames):
        text = f"Frame: {i}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        h, w = frame.shape[:2]
        cv2.putText(frame, text, (w - tw - 10, th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    _step("💾 Saving output video…")
    save_video(out_frames, output_video_path)

    return {
        "n_shots":        len(shot_frames),
        "n_bounces":      len(bounce_frames),
        "shot_frames":    shot_frames,
        "bounce_frames":  bounce_frames,
        "ball_detections": ball_detections,
        "pose_detections": pose_detections,
        "court_keypoints": court_keypoints,
    }


# ---------------------------------------------------------------------------
# Browser-playable re-encode (mp4v → H.264 via ffmpeg)
# ---------------------------------------------------------------------------

def reencode_h264(src: str, dst: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", src,
             "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst],
            capture_output=True, timeout=300,
        )
        return result.returncode == 0 and Path(dst).exists()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Tennis Tracker", page_icon="🎾", layout="wide")

st.title("🎾 Tennis Video Analyser")
st.caption(
    "Upload one or more match clips, set each player's dominant hand, "
    "then click **Run Analysis**."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    uploaded_files = st.file_uploader(
        "Upload video(s)",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        help="Short clips (< 60 s) process fastest. Upload multiple for multi-point analysis.",
    )

    st.divider()
    st.subheader("Player Handedness")
    st.caption("Player 1 = far end (top of frame) · Player 2 = near end (bottom)")

    p1_hand = st.radio("Player 1", ["Right", "Left"], index=0, horizontal=True, key="p1_hand")
    p2_hand = st.radio("Player 2", ["Right", "Left"], index=0, horizontal=True, key="p2_hand")

    st.divider()
    run_btn = st.button(
        "▶ Run Analysis",
        type="primary",
        disabled=(not uploaded_files),
        use_container_width=True,
    )

# ── Main area ─────────────────────────────────────────────────────────────────
if not uploaded_files:
    st.session_state.pop("results", None)
    st.info("👈 Upload one or more videos in the sidebar to get started.")
    st.stop()

# Preview before running
if not run_btn and "results" not in st.session_state:
    st.subheader("Input Preview")
    if len(uploaded_files) == 1:
        st.video(uploaded_files[0])
    else:
        tabs = st.tabs([f"Point {i+1} — {f.name}" for i, f in enumerate(uploaded_files)])
        for tab, uf in zip(tabs, uploaded_files):
            with tab:
                st.video(uf)
    st.stop()

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    tmp_dir = Path(tempfile.mkdtemp())
    clips = []   # will hold per-clip result dicts

    n_clips = len(uploaded_files)
    with st.status(
        f"Running analysis on {n_clips} clip{'s' if n_clips > 1 else ''}…",
        expanded=True,
    ) as status:
        try:
            with _in_project_root():
                for idx, uf in enumerate(uploaded_files, start=1):
                    label = f"Point {idx}"
                    input_path  = str(tmp_dir / f"input_{idx}.mp4")
                    output_path = str(tmp_dir / f"output_raw_{idx}.mp4")
                    output_h264 = str(tmp_dir / f"output_{idx}.mp4")

                    with open(input_path, "wb") as f:
                        f.write(uf.getvalue())

                    result = run_pipeline(
                        input_path,
                        p1_hand.lower(),
                        p2_hand.lower(),
                        output_path,
                        status,
                        clip_label=f"{label} ({uf.name})",
                    )

                    status.update(label=f"[{label}] 🔄 Re-encoding for browser…")
                    if not reencode_h264(output_path, output_h264):
                        output_h264 = output_path

                    clips.append({
                        "point_id":       idx,
                        "label":          label,
                        "filename":       uf.name,
                        "input_path":     input_path,
                        "output_h264":    output_h264,
                        "n_shots":        result["n_shots"],
                        "n_bounces":      result["n_bounces"],
                        # raw data for CSV building
                        "shot_frames":    result["shot_frames"],
                        "bounce_frames":  result["bounce_frames"],
                        "ball_detections": result["ball_detections"],
                        "pose_detections": result["pose_detections"],
                        "court_keypoints": result["court_keypoints"],
                    })

            status.update(label="✅ All clips complete!", state="complete", expanded=False)

            csv_events, csv_ball, csv_pose, csv_court = _build_csvs(clips)

            st.session_state["results"] = {
                "clips":      clips,
                "csv_events": csv_events,
                "csv_ball":   csv_ball,
                "csv_pose":   csv_pose,
                "csv_court":  csv_court,
            }

        except Exception as exc:
            status.update(label=f"❌ Error: {exc}", state="error", expanded=True)
            st.error(str(exc))
            st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.stop()

r = st.session_state["results"]
clips = r["clips"]

# ── Summary metrics ───────────────────────────────────────────────────────────
total_shots   = sum(c["n_shots"]   for c in clips)
total_bounces = sum(c["n_bounces"] for c in clips)
col1, col2, col3 = st.columns(3)
col1.metric("Total shots detected",   total_shots)
col2.metric("Total bounces detected", total_bounces)
col3.metric("Points analysed",        len(clips))

st.divider()

tab_results, tab_dash = st.tabs(["📹 Videos & Downloads", "🎾 Serve Dashboard"])

# ── Tab 1: Videos & CSV downloads ─────────────────────────────────────────────
with tab_results:
    tab_labels = [f"Point {c['point_id']} — {c['filename']}" for c in clips]

    col_in, col_out = st.columns(2)

    with col_in:
        st.subheader("Input")
        if len(clips) == 1:
            st.video(clips[0]["input_path"])
        else:
            for tab, clip in zip(st.tabs(tab_labels), clips):
                with tab:
                    st.caption(f"{clip['n_shots']} shot(s) · {clip['n_bounces']} bounce(s)")
                    st.video(clip["input_path"])

    with col_out:
        st.subheader("Output (annotated)")
        if len(clips) == 1:
            st.video(clips[0]["output_h264"])
        else:
            for tab, clip in zip(st.tabs(tab_labels), clips):
                with tab:
                    st.caption(f"{clip['n_shots']} shot(s) · {clip['n_bounces']} bounce(s)")
                    st.video(clip["output_h264"])

    st.divider()
    st.subheader("📥 Download CSVs")
    dl1, dl2, dl3, dl4 = st.columns(4)
    dl1.download_button("⬇ events.csv",          r["csv_events"], "events.csv",          "text/csv", use_container_width=True)
    dl2.download_button("⬇ ball_coords.csv",     r["csv_ball"],   "ball_coords.csv",     "text/csv", use_container_width=True)
    dl3.download_button("⬇ pose_joints.csv",     r["csv_pose"],   "pose_joints.csv",     "text/csv", use_container_width=True)
    dl4.download_button("⬇ court_keypoints.csv", r["csv_court"],  "court_keypoints.csv", "text/csv", use_container_width=True)

# ── Tab 2: Serve Dashboard ────────────────────────────────────────────────────
with tab_dash:
    df_court_mem  = pd.read_csv(io.StringIO(r["csv_court"]))
    df_ball_mem   = pd.read_csv(io.StringIO(r["csv_ball"]))
    df_events_mem = pd.read_csv(io.StringIO(r["csv_events"]))

    is_multi_dash = "point" in df_court_mem.columns
    fc1, fc2, fc3, fc4 = st.columns(4)

    if is_multi_dash:
        all_pts = sorted(df_court_mem["point"].unique())
        dash_pts = fc1.multiselect("Point", all_pts, default=all_pts,
                                   format_func=lambda p: f"Point {p}", key="dash_pts")
    else:
        dash_pts = None
        fc1.empty()

    dash_players = fc2.multiselect("Player", ["player 1", "player 2"],
                                   default=["player 1", "player 2"],
                                   format_func=lambda p: p.title(), key="dash_players")
    dash_sides   = fc3.multiselect("Court side", ["Deuce", "Ad"],
                                   default=["Deuce", "Ad"], key="dash_sides")

    fig = build_serve_figure(
        df_court_mem, df_ball_mem, df_events_mem,
        selected_players=dash_players,
        selected_sides=dash_sides,
        selected_points=dash_pts,
    )
    st.plotly_chart(fig, use_container_width=True)
