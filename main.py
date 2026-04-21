import numpy as np
from court_line_detector.court_line_detector import CourtLineDetector
from utils import read_video, save_video, draw_axes
from trackers import PlayerTracker, PoseDetector, BallTracker, ShotTracker
import argparse
import csv

def make_blank_frames(video_frames):
    """Return a list of black frames matching the shape of the input frames."""
    return [np.zeros_like(frame) for frame in video_frames]

def save_pose_csv(pose_detections, output_csv="pose_output.csv"):
    """
    Writes CSV with columns:
    frame, joint, player1_x, player1_y, player2_x, player2_y
    """
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame",
            "joint",
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

                writer.writerow([
                    frame_idx,
                    joint_idx,
                    p1x,
                    p1y,
                    p2x,
                    p2y
                ])
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
    args = parser.parse_args()
 
    input_video_path = "input_videos/clip.mp4"
    video_frames = read_video(input_video_path)
 
    # Court
    court_line_detector = CourtLineDetector("keypoints_model.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])
 
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
    #shot_tracker = ShotTracker(
    #    minimum_change_frames=25,   # frames the vertical direction change must persist
    #    rolling_window=5,           # smoothing window for mid_y
    #    wrist_proximity_px=160.0,   # px radius around ball to check for joints
    #    persist_frames=20,          # circle stays visible for 20 frames
    #)
    #shot_frames = shot_tracker.detect_shots(ball_detections, pose_detections)
    #print(f"Detected {len(shot_frames)} shot(s) at frames: {sorted(shot_frames.keys())}")
 
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

    # Draw shot markers (on top of everything else so they're clearly visible)
    #output_video_frames = shot_tracker.draw_shot_markers(output_video_frames, shot_frames)

    output_video_frames = pose_detector.draw_poses(output_video_frames, pose_detections)
 
    output_path = (
        "output_videos/output_annotations_only.mp4"
        if args.annotations_only
        else "output_videos/output.mp4"
    )
    save_video(output_video_frames, output_path)
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    main()