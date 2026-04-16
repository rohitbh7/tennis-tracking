import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance


# COCO keypoint indices
CONTACT_JOINTS = [7, 8, 9, 10]  # left_elbow, right_elbow, left_wrist, right_wrist


class ShotTracker:
    """
    Detects tennis shots by combining two signals:

      1. Ball trajectory (delta_y) — a real shot causes a sustained reversal in
         the ball's vertical direction for at least `minimum_change_frames` frames,
         distinguishing true hits from noise blips.

      2. Wrist proximity — at the candidate frame, at least one wrist/elbow joint
         must be within `wrist_proximity_px` pixels of the ball center, filtering
         out bounces (which also reverse vertical direction but have no nearby player).

    Usage
    -----
        tracker = ShotTracker()
        shot_frames = tracker.detect_shots(ball_positions, pose_detections)
        annotated   = tracker.draw_shot_markers(frames, shot_frames)
    """

    def __init__(
        self,
        minimum_change_frames: int = 25,    # frames the direction change must persist
        rolling_window: int = 5,            # smoothing window for mid_y
        wrist_proximity_px: float = 160.0,  # max px from ball to wrist/elbow
        marker_color: tuple = (0, 255, 255),
        marker_radius: int = 18,
        marker_thickness: int = 3,
        persist_frames: int = 20,
    ):
        self.minimum_change_frames = minimum_change_frames
        self.rolling_window = rolling_window
        self.wrist_proximity_px = wrist_proximity_px
        self.marker_color = marker_color
        self.marker_radius = marker_radius
        self.marker_thickness = marker_thickness
        self.persist_frames = persist_frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_shots(
        self,
        ball_positions: list,
        pose_detections: list,
    ) -> dict[int, tuple]:
        """
        Parameters
        ----------
        ball_positions  : list of {1: [x1, y1, x2, y2]} dicts, one per frame
                          (the standard format from BallTracker.detect_frames)
        pose_detections : list of sv.KeyPoints | None, one per frame

        Returns
        -------
        shot_frames : dict mapping frame_index -> (ball_cx, ball_cy)
        """
        ball_hit_frames = self._get_ball_shot_frames(ball_positions)

        shot_frames = {}
        for frame_idx in ball_hit_frames:
            pos = ball_positions[frame_idx]

            if isinstance(pos, dict):
                bbox = pos.get(1, [])
                if not bbox or len(bbox) < 4:
                    continue
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
            else:
                cx, cy = pos
                if cx is None or cy is None:
                    continue
                cx, cy = int(cx), int(cy)

            if self._player_near_ball(pose_detections, frame_idx, cx, cy):
                shot_frames[frame_idx] = (cx, cy)

        return shot_frames

    def draw_shot_markers(
        self,
        frames: list,
        shot_frames: dict[int, tuple],
    ) -> list:
        """
        Draw a fading circle at each shot location for `persist_frames` frames.
        """
        output_frames = []
        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()

            for shot_idx, (bx, by) in shot_frames.items():
                if shot_idx <= frame_idx < shot_idx + self.persist_frames:
                    age = frame_idx - shot_idx
                    alpha = 1.0 - age / self.persist_frames

                    color = tuple(int(c * alpha) for c in self.marker_color)
                    cv2.circle(annotated, (bx, by), self.marker_radius, color, self.marker_thickness)

                    if age == 0:
                        cv2.circle(annotated, (bx, by), 4, self.marker_color, -1)

                    if age < 8:
                        label_alpha = 1.0 - age / 8
                        label_color = tuple(int(c * label_alpha) for c in self.marker_color)
                        cv2.putText(
                            annotated, "SHOT",
                            (bx + self.marker_radius + 4, by + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA,
                        )

            output_frames.append(annotated)

        return output_frames

    # ------------------------------------------------------------------
    # Ball trajectory logic (delta_y sustained change)
    # ------------------------------------------------------------------

    def _get_ball_shot_frames(self, ball_positions: list) -> list[int]:
        """
        Identify frames where the ball's vertical direction reverses and that
        reversal is sustained for at least `minimum_change_frames` frames.

        Adapted from: https://github.com/abdullahtarek/tennis_analysis
        """
        # Support both formats:
        #   TrackNet BallTracker  → list of (cx, cy) tuples
        #   YOLO BallTracker      → list of {1: [x1, y1, x2, y2]} dicts
        if ball_positions and isinstance(ball_positions[0], dict):
            raw = [x.get(1, []) for x in ball_positions]
            df = pd.DataFrame(raw, columns=['x1', 'y1', 'x2', 'y2'])
        else:
            # (cx, cy) or (None, None) — synthesise a 1-pixel bbox around the centre
            rows = []
            for pos in ball_positions:
                cx, cy = pos
                if cx is None or cy is None:
                    rows.append([np.nan, np.nan, np.nan, np.nan])
                else:
                    rows.append([cx - 1, cy - 1, cx + 1, cy + 1])
            df = pd.DataFrame(rows, columns=['x1', 'y1', 'x2', 'y2'])

        df['ball_hit'] = 0
        df['mid_y'] = (df['y1'] + df['y2']) / 2

        # Smooth vertical position to suppress jitter
        df['mid_y_rolling'] = (
            df['mid_y']
            .rolling(window=self.rolling_window, min_periods=1, center=False)
            .mean()
        )

        # Frame-to-frame vertical velocity (positive = moving down, negative = up)
        df['delta_y'] = df['mid_y_rolling'].diff()

        lookahead = int(self.minimum_change_frames * 1.2)

        for i in range(1, len(df) - lookahead):
            dy_i = df['delta_y'].iloc[i]
            dy_next = df['delta_y'].iloc[i + 1]

            negative_change = dy_i > 0 and dy_next < 0  # downward -> upward
            positive_change = dy_i < 0 and dy_next > 0  # upward -> downward

            if not (negative_change or positive_change):
                continue

            # Count how many of the following frames sustain the same reversal
            change_count = 0
            for j in range(i + 1, i + lookahead + 1):
                dy_j = df['delta_y'].iloc[j]
                if negative_change and dy_i > 0 and dy_j < 0:
                    change_count += 1
                elif positive_change and dy_i < 0 and dy_j > 0:
                    change_count += 1

            if change_count >= self.minimum_change_frames:
                df.loc[i, 'ball_hit'] = 1

        return df[df['ball_hit'] == 1].index.tolist()

    # ------------------------------------------------------------------
    # Pose proximity check
    # ------------------------------------------------------------------

    def _player_near_ball(
        self,
        pose_detections: list,
        frame_idx: int,
        bx: float,
        by: float,
    ) -> bool:
        """
        Return True if any wrist or elbow joint is within `wrist_proximity_px`
        pixels of (bx, by) at exactly frame_idx.
        """
        kp = pose_detections[frame_idx]
        if kp is None:
            return False

        for player_joints in kp.xy:   # shape: (num_players, num_joints, 2)
            for joint_idx in CONTACT_JOINTS:
                if joint_idx >= len(player_joints):
                    continue
                jx, jy = player_joints[joint_idx]
                if jx == 0 and jy == 0:  # joint not detected
                    continue
                if distance.euclidean((bx, by), (jx, jy)) <= self.wrist_proximity_px:
                    return True

        return False