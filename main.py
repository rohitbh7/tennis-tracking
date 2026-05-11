import numpy as np
from court_line_detector.court_line_detector import CourtLineDetector
from utils import read_video, save_video, draw_axes
from trackers import PlayerTracker, PoseDetector, BallTracker, ShotTracker, BounceTracker
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

        x = w - tw - 10   # right padding
        y = th + 10       # top padding

        cv2.putText(
            f,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),   # white text
            thickness,
            cv2.LINE_AA
        )

        output.append(f)

    return output

def make_blank_frames(video_frames):
    """Return a list of black frames matching the shape of the input frames."""
    return [np.zeros_like(frame) for frame in video_frames]

_COCO_JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

def save_pose_csv(pose_detections, output_csv="pose_output.csv"):
    """
    Writes CSV with columns:
    frame, joint, joint_name, player1_x, player1_y, player2_x, player2_y
    """
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "joint",
            "joint_name",
            "player1_x",
            "player1_y",
            "player2_x",
            "player2_y"
        ])

        for frame_idx, kp in enumerate(pose_detections):
            if kp is None:
                continue

            # shape: (num_players, num_joints, 2)
            xy = kp.xy

            # require two players
            if xy.shape[0] < 2:
                continue

            player1 = xy[0]
            player2 = xy[1]

            num_joints = min(len(player1), len(player2))

            for joint_idx in range(num_joints):
                p1x, p1y = player1[joint_idx]
                p2x, p2y = player2[joint_idx]
                joint_name = _COCO_JOINT_NAMES[joint_idx] if joint_idx < len(_COCO_JOINT_NAMES) else str(joint_idx)

                writer.writerow([
                    frame_idx,
                    joint_idx,
                    joint_name,
                    p1x,
                    p1y,
                    p2x,
                    p2y
                ])
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

def save_court_keypoints_csv(court_keypoints, output_csv="court_keypoints.csv"):
    """
    Writes CSV with columns:
    keypoint_id, court_location, x, y
    """
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keypoint_id", "court_location", "x", "y"])
        for i in range(14):
            x = court_keypoints[i * 2]
            y = court_keypoints[i * 2 + 1]
            writer.writerow([i, _COURT_KEYPOINT_NAMES[i], x, y])

def save_events_csv(shot_frames, bounce_frames, output_csv="events.csv"):
    """
    Writes CSV with columns:
    frame, event, player

    shot_frames : dict  frame -> (cx, cy, label, player_num)
    bounce_frames: dict frame -> (cx, cy)
    """
    rows = []

    for frame, data in shot_frames.items():
        label = data[2]
        player_num = data[3]
        player = f"player {player_num}" if player_num is not None else ""
        rows.append((frame, label, player))

    for frame in bounce_frames:
        rows.append((frame, "bounce", ""))

    rows.sort(key=lambda r: r[0])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "event", "player"])
        writer.writerows(rows)

def save_ball_csv(ball_detections, output_csv="ball_coords.csv"):
    """
    Writes CSV with columns:
    frame, ball_x, ball_y
    """
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "ball_x", "ball_y"])

        for frame_idx, (x, y) in enumerate(ball_detections):
            writer.writerow([frame_idx, x, y])

def main():
    parser = argparse.ArgumentParser(description="Tennis video analysis pipeline.")
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
 
    input_video_path = "input_videos/sinner_alcaraz_point.mp4"
    video_frames = read_video(input_video_path)
 
    # Court
    court_line_detector = CourtLineDetector("keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])
    save_court_keypoints_csv(court_keypoints, "output_videos/court_keypoints.csv")
 
    # Players
    player_tracker = PlayerTracker(model_path="yolo12n.pt")
    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=False, stub_path="tracker_stubs/player_detection.pkl"
    )
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    
    # Poses
    pose_detector = PoseDetector()
    pose_detections = pose_detector.detect_frames(video_frames, player_detections=player_detections)
    save_pose_csv(pose_detections, "output_videos/pose_joints.csv")
 
    
    # Ball
    ball_tracker = BallTracker(model_path="model_best.pt", device='cpu')
    ball_detections = ball_tracker.detect_frames(video_frames, extrapolation=True)
    save_ball_csv(ball_detections, "output_videos/ball_coords.csv")

    # Shots
    shot_tracker = ShotTracker(
        minimum_change_frames=15,   # frames the vertical direction change must persist
        rolling_window=5,           # smoothing window for mid_y
        wrist_proximity_px=125.0,   # px radius around ball to check for joints
        persist_frames=20,          # circle stays visible for 20 frames
        p1_handedness=args.p1_hand,
        p2_handedness=args.p2_hand,
    )
    frame_height, frame_width = video_frames[0].shape[:2]
    shot_frames = shot_tracker.detect_shots(
        ball_detections, pose_detections,
        frame_height=frame_height,
        frame_width=frame_width,
    )
    print(f"Detected {len(shot_frames)} shot(s) at frames: {sorted(shot_frames.keys())}")

    # Bounces
    bounce_tracker = BounceTracker()
    bounce_frames = bounce_tracker.detect_bounces(
        ball_detections, shot_frames=shot_frames, pose_detections=pose_detections
    )
    print(f"Detected {len(bounce_frames)} bounce(s) at frames: {sorted(bounce_frames.keys())}")
    save_events_csv(shot_frames, bounce_frames, "output_videos/events.csv")

    # Use blank frames as the canvas if --annotations-only is set
    canvas_frames = make_blank_frames(video_frames) if args.annotations_only else video_frames
 
    # Draw everything
    output_video_frames = player_tracker.draw_bboxes(canvas_frames, player_detections)
 
    if args.annotations_only:
        # Draw clean court lines instead of raw keypoint dots
        output_video_frames = [
            court_line_detector.draw_lines(f, court_keypoints) for f in output_video_frames
        ]
    else:
        output_video_frames = [
            court_line_detector.draw_keypoints(f, court_keypoints) for f in output_video_frames
        ]
 
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    output_video_frames = bounce_tracker.draw_bounces(output_video_frames, bounce_frames)

    # Draw shot markers (on top of everything else so they're clearly visible)
    output_video_frames = shot_tracker.draw_shot_markers(output_video_frames, shot_frames)

    output_video_frames = pose_detector.draw_poses(output_video_frames, pose_detections)
 
    output_path = (
        "output_videos/output_annotations_only.mp4"
        if args.annotations_only
        else "output_videos/output.mp4"
    )
    output_video_frames = draw_frame_numbers(output_video_frames)
    save_video(output_video_frames, output_path)
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    main()