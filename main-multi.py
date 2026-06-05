import os
import numpy as np
from court_line_detector.court_line_detector import CourtLineDetector
from utils import read_video, save_video, draw_axes
from trackers import PlayerTracker, PoseDetector, BallTracker, ShotTracker2, BounceTracker
import argparse
import csv
import cv2

def draw_frame_numbers(frames):
    output = []
    for i, frame in enumerate(frames):
        f = frame.copy()
        h, w = f.shape[:2]

        text = f"Frame: {i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2

        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

        x = w - tw - 10
        y = th + 10

        cv2.putText(
            f,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        output.append(f)

    return output

def make_blank_frames(video_frames):
    return [np.zeros_like(frame) for frame in video_frames]

_COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

_COURT_KEYPOINT_NAMES = [
    "back-left baseline corner",
    "back-right baseline corner",
    "front-left baseline corner",
    "front-right baseline corner",
    "left alley back",
    "left alley front",
    "right alley back",
    "right alley front",
    "back service line left",
    "back service line right",
    "front service line left",
    "front service line right",
    "service box centre back",
    "service box centre front",
]


def collect_pose_rows(pose_detections, clip_id):
    rows = []
    for frame_idx, kp in enumerate(pose_detections):
        if kp is None:
            continue
        xy = kp.xy
        if xy.shape[0] < 2:
            continue
        player1 = xy[0]
        player2 = xy[1]
        num_joints = min(len(player1), len(player2))
        for joint_idx in range(num_joints):
            p1x, p1y = player1[joint_idx]
            p2x, p2y = player2[joint_idx]
            joint_name = _COCO_JOINT_NAMES[joint_idx] if joint_idx < len(_COCO_JOINT_NAMES) else str(joint_idx)
            rows.append([clip_id, frame_idx, joint_idx, joint_name, p1x, p1y, p2x, p2y])
    return rows


def collect_court_keypoint_rows(court_keypoints, clip_id):
    rows = []
    for i in range(14):
        x = court_keypoints[i * 2]
        y = court_keypoints[i * 2 + 1]
        rows.append([clip_id, i, _COURT_KEYPOINT_NAMES[i], x, y])
    return rows


def collect_events_rows(shot_frames, bounce_frames, clip_id):
    rows = []
    for frame, data in shot_frames.items():
        label = data[2]
        player_num = data[3]
        player = f"player {player_num}" if player_num is not None else ""
        rows.append([clip_id, frame, label, player])
    for frame in bounce_frames:
        rows.append([clip_id, frame, "bounce", ""])
    rows.sort(key=lambda r: r[1])
    return rows


def collect_ball_rows(ball_detections, clip_id):
    rows = []
    for frame_idx, (x, y) in enumerate(ball_detections):
        rows.append([clip_id, frame_idx, x, y])
    return rows


def save_pose_csv(all_rows, output_csv="output_videos/pose_joints.csv"):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "frame", "joint", "joint_name", "player1_x", "player1_y", "player2_x", "player2_y"])
        writer.writerows(all_rows)


def save_court_keypoints_csv(all_rows, output_csv="output_videos/court_keypoints.csv"):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "keypoint_id", "court_location", "x", "y"])
        writer.writerows(all_rows)


def save_events_csv(all_rows, output_csv="output_videos/events.csv"):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "frame", "event", "player"])
        writer.writerows(all_rows)


def save_ball_csv(all_rows, output_csv="output_videos/ball_coords.csv"):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "frame", "ball_x", "ball_y"])
        writer.writerows(all_rows)


def process_clip(clip_path, clip_id, args):
    print(f"\n=== Processing clip {clip_id}: {clip_path} ===")
    video_frames = read_video(clip_path)

    court_line_detector = CourtLineDetector("keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    player_tracker = PlayerTracker(model_path="yolo12n.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False, stub_path=None)
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    pose_detector = PoseDetector()
    pose_detections = pose_detector.detect_frames(video_frames, player_detections=player_detections)

    ball_tracker = BallTracker(model_path="model_best.pt", device='cpu')
    ball_detections = ball_tracker.detect_frames(video_frames, extrapolation=True)

    shot_tracker = ShotTracker2(
        wrist_proximity_px=100.0,
        min_velocity_change=12.0,
        min_post_speed=7.0,
        persist_frames=20,
        p1_handedness=args.p1_hand,
        p2_handedness=args.p2_hand,
    )
    frame_height, frame_width = video_frames[0].shape[:2]
    shot_frames = shot_tracker.detect_shots(
        ball_detections, pose_detections,
        frame_height=frame_height,
        frame_width=frame_width,
    )
    print(f"  Clip {clip_id}: {len(shot_frames)} shot(s) at frames: {sorted(shot_frames.keys())}")

    bounce_tracker = BounceTracker()
    bounce_frames = bounce_tracker.detect_bounces(
        ball_detections, shot_frames=shot_frames, pose_detections=pose_detections
    )
    print(f"  Clip {clip_id}: {len(bounce_frames)} bounce(s) at frames: {sorted(bounce_frames.keys())}")

    canvas_frames = make_blank_frames(video_frames) if args.annotations_only else video_frames

    output_video_frames = player_tracker.draw_bboxes(canvas_frames, player_detections)

    if args.annotations_only:
        output_video_frames = [
            court_line_detector.draw_lines(f, court_keypoints) for f in output_video_frames
        ]
    else:
        output_video_frames = [
            court_line_detector.draw_keypoints(f, court_keypoints) for f in output_video_frames
        ]

    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = bounce_tracker.draw_bounces(output_video_frames, bounce_frames)
    output_video_frames = shot_tracker.draw_shot_markers(output_video_frames, shot_frames)
    output_video_frames = pose_detector.draw_poses(output_video_frames, pose_detections)
    output_video_frames = draw_frame_numbers(output_video_frames)

    clip_stem = os.path.splitext(os.path.basename(clip_path))[0]
    suffix = "_annotations_only" if args.annotations_only else ""
    output_path = f"output_videos/{clip_stem}{suffix}.mp4"
    save_video(output_video_frames, output_path)
    print(f"  Clip {clip_id}: saved to {output_path}")

    pose_rows = collect_pose_rows(pose_detections, clip_id)
    court_rows = collect_court_keypoint_rows(court_keypoints, clip_id)
    events_rows = collect_events_rows(shot_frames, bounce_frames, clip_id)
    ball_rows = collect_ball_rows(ball_detections, clip_id)

    return pose_rows, court_rows, events_rows, ball_rows


def main():
    parser = argparse.ArgumentParser(description="Tennis video analysis pipeline — multi-clip.")
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help=(
            "Render annotations on a black background instead of the original "
            "video frames. Court lines are drawn instead of raw keypoint dots."
        ),
    )
    parser.add_argument(
        "p1_hand",
        choices=["right", "left"],
        help="Player 1 handedness (top of screen).",
    )
    parser.add_argument(
        "p2_hand",
        choices=["right", "left"],
        help="Player 2 handedness (bottom of screen).",
    )
    args = parser.parse_args()

    input_folder = "input_videos/game"

    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    clip_paths = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in video_extensions
    ])

    if not clip_paths:
        print(f"No video files found in {input_folder}")
        return

    print(f"Found {len(clip_paths)} clip(s) in {input_folder}")

    all_pose_rows = []
    all_court_rows = []
    all_events_rows = []
    all_ball_rows = []

    for clip_id, clip_path in enumerate(clip_paths, start=1):
        pose_rows, court_rows, events_rows, ball_rows = process_clip(clip_path, clip_id, args)
        all_pose_rows.extend(pose_rows)
        all_court_rows.extend(court_rows)
        all_events_rows.extend(events_rows)
        all_ball_rows.extend(ball_rows)

    save_pose_csv(all_pose_rows)
    save_court_keypoints_csv(all_court_rows)
    save_events_csv(all_events_rows)
    save_ball_csv(all_ball_rows)

    print(f"\nDone! Processed {len(clip_paths)} clip(s). CSVs saved to output_videos/")


if __name__ == "__main__":
    main()
