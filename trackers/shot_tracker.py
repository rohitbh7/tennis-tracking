import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance


# COCO keypoint indices
CONTACT_JOINTS = [7, 8, 9, 10]  # left_elbow, right_elbow, left_wrist, right_wrist

# COCO keypoint indices used for shot classification
_NOSE        = 0
_L_SHOULDER  = 5
_R_SHOULDER  = 6
_L_HIP       = 11
_R_HIP       = 12
_L_ANKLE     = 15
_R_ANKLE     = 16

# Joints used to find the player's horizontal centre (torso midline)
_CENTRE_JOINTS = [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP]

# Joints used to estimate ankle height (serve vs smash gate)
_ANKLE_JOINTS = [_L_ANKLE, _R_ANKLE]


class ShotTracker:
    """
    Detects tennis shots by combining three signals:

      1. Ball trajectory (delta_y) — a real shot causes a sustained reversal in
         the ball's vertical direction for at least `minimum_change_frames` frames,
         distinguishing true hits from noise blips.

      2. Ball trajectory (delta_x) — same sustained-reversal logic applied
         horizontally, catching flat shots that barely change vertical direction.

      3. Wrist closing velocity — finds frames where a wrist/elbow reaches its
         closest approach to the ball (the derivative of wrist-ball distance crosses
         zero). This is direction-agnostic and physically corresponds to the moment
         of contact regardless of shot type.

    Signals 1 & 2 are the primary detectors. Signal 3 plays two roles:

      - Precision refinement: if a closing velocity candidate falls within
        `dedup_window` frames of a trajectory candidate, it replaces the trajectory
        frame as the more precise contact timestamp (correcting rolling-average lag
        without hardcoding a fixed -1 offset).

      - Standalone fallback: if a closing velocity candidate has no nearby trajectory
        signal (e.g. a flat shot with no vertical/horizontal reversal), it is accepted
        only if the wrist-ball distance is below the tighter `closing_velocity_standalone_px`
        threshold, preventing premature triggers during the player's approach swing.

    Shot classification
    -------------------
    Each detected hit is labelled FOREHAND, BACKHAND, SMASH, or SERVE.

    Player layout (standard broadcast view):
      • Player 1 occupies the **top** half of the frame.
      • Player 2 occupies the **bottom** half of the frame.

    Handedness rules for Player 1 (top of screen, facing downward):
      • Righty → ball left of body = FOREHAND, ball right = BACKHAND
      • Lefty  → ball left of body = BACKHAND,  ball right = FOREHAND

    Handedness rules for Player 2 (bottom of screen, facing upward — mirrored):
      • Righty → ball right of body = FOREHAND, ball left = BACKHAND
      • Lefty  → ball right of body = BACKHAND,  ball left = FOREHAND

    Overhead shots (ball above the player's head):
      • SERVE  — ankles sit near their baseline (upper ~35 % of frame for P1,
                 lower ~35 % for P2), meaning the player hasn't moved in yet.
      • SMASH  — ankles are further inside the court.

    Usage
    -----
        tracker = ShotTracker(p1_handedness='right', p2_handedness='left')
        shot_frames = tracker.detect_shots(ball_positions, pose_detections,
                                           frame_height=720, frame_width=1280)
        annotated   = tracker.draw_shot_markers(frames, shot_frames)
    """

    def __init__(
        self,
        minimum_change_frames: int = 15,     # frames the direction change must persist (signals 1 & 2)
        rolling_window: int = 5,             # smoothing window for mid_y / mid_x
        wrist_proximity_px: float = 160.0,   # max px from ball to wrist/elbow (all signals)
        closing_velocity_px: float = 50.0,            # max dist to register a closing-velocity local minimum
        closing_velocity_standalone_px: float = 10.0, # tighter threshold when no trajectory signal nearby
        dedup_window: int = 10,                       # frames within which candidates are merged
        marker_color: tuple = (0, 255, 255),
        marker_radius: int = 18,
        marker_thickness: int = 3,
        persist_frames: int = 20,
        # --- Handedness ('right' or 'left') ---
        p1_handedness: str = 'right',
        p2_handedness: str = 'right',
        # Fraction of frame height within which ankles must fall to count as "at baseline".
        # Player 1's ankles must be above  baseline_fraction * frame_height.
        # Player 2's ankles must be below (1 - baseline_fraction) * frame_height.
        baseline_fraction: float = 0.35,
    ):
        self.minimum_change_frames = minimum_change_frames
        self.rolling_window = rolling_window
        self.wrist_proximity_px = wrist_proximity_px
        self.closing_velocity_px = closing_velocity_px
        self.closing_velocity_standalone_px = closing_velocity_standalone_px
        self.dedup_window = dedup_window
        self.marker_color = marker_color
        self.marker_radius = marker_radius
        self.marker_thickness = marker_thickness
        self.persist_frames = persist_frames
        self.p1_handedness = p1_handedness.lower()
        self.p2_handedness = p2_handedness.lower()
        self.baseline_fraction = baseline_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_shots(
        self,
        ball_positions: list,
        pose_detections: list,
        frame_height: int = 720,
        frame_width: int = 1280,
    ) -> dict[int, tuple]:
        """
        Parameters
        ----------
        ball_positions  : list of {1: [x1, y1, x2, y2]} dicts, one per frame
                          (the standard format from BallTracker.detect_frames)
        pose_detections : list of sv.KeyPoints | None, one per frame
        frame_height    : pixel height of the video frames (used for shot classification)
        frame_width     : pixel width  of the video frames (used for shot classification)

        Returns
        -------
        shot_frames : dict mapping frame_index -> (ball_cx, ball_cy, shot_label)
                      shot_label is one of: 'FOREHAND', 'BACKHAND', 'SMASH', 'SERVE'
        """
        # --- Gather candidates from trajectory signals ---
        traj_y_frames = self._get_trajectory_shot_frames(ball_positions, axis='y')
        traj_x_frames = self._get_trajectory_shot_frames(ball_positions, axis='x')
        closing_frames, closing_dists = self._get_closing_velocity_frames(ball_positions, pose_detections)

        # Apply rolling-average lag correction to trajectory candidates
        traj_y_frames = [max(0, f - 1) for f in traj_y_frames]
        traj_x_frames = [max(0, f - 1) for f in traj_x_frames]
        traj_frames = sorted(set(traj_y_frames + traj_x_frames))

        # --- Merge closing velocity candidates with trajectory candidates ---
        # Each closing velocity candidate either refines a nearby trajectory frame
        # (replacing it with the more precise contact timestamp) or is accepted
        # standalone only if the wrist-ball distance is below the tighter threshold.
        consumed_traj = set()
        refined = {}   # traj_frame -> closing_frame replacement
        standalone_closing = []

        for cv_frame, cv_dist in zip(closing_frames, closing_dists):
            nearby_traj = [
                f for f in traj_frames
                if abs(cv_frame - f) < self.dedup_window and f not in consumed_traj
            ]
            if nearby_traj:
                closest = min(nearby_traj, key=lambda f: abs(cv_frame - f))
                if cv_frame >= closest:
                    # Closing vel fires at or after the traj candidate — use it as the
                    # more precise contact timestamp (corrects rolling-average lag)
                    refined[closest] = cv_frame
                    consumed_traj.add(closest)
                elif cv_dist <= self.closing_velocity_standalone_px:
                    # Closing vel fires before the traj candidate but is extremely close —
                    # treat as standalone (e.g. approach local min that is genuinely contact)
                    standalone_closing.append(cv_frame)
                # else: closing vel fires earlier and isn't tight enough — ignore it,
                # the trajectory candidate is more reliable
            elif cv_dist <= self.closing_velocity_standalone_px:
                # No trajectory signal nearby — only accept if extremely close
                standalone_closing.append(cv_frame)

        # Build final candidate list: refined traj frames + unrefinied traj frames + standalone closing
        all_candidates = []
        for f in traj_frames:
            all_candidates.append(refined.get(f, f))
        all_candidates.extend(standalone_closing)
        all_candidates = sorted(set(all_candidates))

        # --- Deduplicate candidates within dedup_window frames ---
        deduped = []
        for frame_idx in all_candidates:
            if deduped and (frame_idx - deduped[-1]) < self.dedup_window:
                continue
            deduped.append(frame_idx)

        # --- Gate each candidate with wrist proximity and classify ---
        shot_frames = {}
        for frame_idx in deduped:
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
                label = self._classify_shot(
                    frame_idx, cx, cy,
                    pose_detections, frame_height, frame_width,
                )
                shot_frames[frame_idx] = (cx, cy, label)

        return shot_frames

    def draw_shot_markers(
        self,
        frames: list,
        shot_frames: dict[int, tuple],
    ) -> list:
        """
        Draw a fading circle and shot-type label at each shot location
        for `persist_frames` frames.

        shot_frames values are expected to be (ball_cx, ball_cy, shot_label)
        as returned by detect_shots().
        """
        # Label colours per shot type for quick visual differentiation
        _label_colors = {
            'FOREHAND': (0, 255, 100),   # green-ish
            'BACKHAND': (100, 180, 255), # blue-ish
            'SMASH':    (0, 80, 255),    # red-orange
            'SERVE':    (255, 200, 0),   # gold
        }

        output_frames = []
        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()

            for shot_idx, shot_data in shot_frames.items():
                bx, by = shot_data[0], shot_data[1]
                label  = shot_data[2] if len(shot_data) > 2 else 'SHOT'

                if shot_idx <= frame_idx < shot_idx + self.persist_frames:
                    age = frame_idx - shot_idx
                    alpha = 1.0 - age / self.persist_frames

                    base_color = _label_colors.get(label, self.marker_color)
                    color = tuple(int(c * alpha) for c in base_color)
                    cv2.circle(annotated, (bx, by), self.marker_radius, color, self.marker_thickness)

                    if age == 0:
                        cv2.circle(annotated, (bx, by), 4, base_color, -1)

                    if age < 8:
                        label_alpha = 1.0 - age / 8
                        label_color = tuple(int(c * label_alpha) for c in base_color)
                        cv2.putText(
                            annotated, label,
                            (bx + self.marker_radius + 4, by + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA,
                        )

            output_frames.append(annotated)

        return output_frames

    # ------------------------------------------------------------------
    # Shot classification
    # ------------------------------------------------------------------

    def _classify_shot(
        self,
        frame_idx: int,
        ball_x: float,
        ball_y: float,
        pose_detections: list,
        frame_height: int,
        frame_width: int,
    ) -> str:
        """
        Return one of 'FOREHAND', 'BACKHAND', 'SMASH', or 'SERVE'.

        Steps
        -----
        1. Split detected players into top-half (Player 1) and bottom-half
           (Player 2) by the average y of their visible keypoints.
        2. Decide which player hit the shot: whichever half the ball is in.
        3. Compute the player's torso midline x (shoulders + hips average).
        4. If the ball is above the player's head → overhead shot:
             - ankles near their own baseline → SERVE
             - otherwise → SMASH
        5. Otherwise → FOREHAND or BACKHAND based on ball side × handedness.
        """
        kp = pose_detections[frame_idx] if frame_idx < len(pose_detections) else None
        if kp is None or len(kp.xy) == 0:
            return 'SHOT'

        # --- 1. Find the hitter by contact-joint proximity to the ball ---
        # This is robust regardless of frame resolution: we don't use ball_y vs
        # frame_height/2 (which breaks when the ball is between the two players).
        # Instead, whoever's wrist/elbow is closest to the ball is the hitter —
        # consistent with the _player_near_ball gate that already confirmed contact.
        best_dist   = np.inf
        hitting_player = None
        hitting_avg_y  = None

        for player_joints in kp.xy:
            # Contact-joint distance to ball for this player
            for ji in CONTACT_JOINTS:
                if ji >= len(player_joints):
                    continue
                jx, jy = player_joints[ji]
                if jx == 0 and jy == 0:
                    continue
                d = distance.euclidean((ball_x, ball_y), (jx, jy))
                if d < best_dist:
                    best_dist      = d
                    hitting_player = player_joints
                    # Average y of this player's visible keypoints (determines P1 vs P2)
                    valid_ys = [
                        player_joints[j][1]
                        for j in range(len(player_joints))
                        if not (player_joints[j][0] == 0 and player_joints[j][1] == 0)
                    ]
                    hitting_avg_y = float(np.mean(valid_ys)) if valid_ys else None

        if hitting_player is None or hitting_avg_y is None:
            return 'SHOT'

        # --- 2. Determine player number and handedness from the hitter's body position ---
        # Player 1 occupies the top half of the frame; Player 2 the bottom.
        if hitting_avg_y < frame_height / 2:
            player_num = 1
            handedness = self.p1_handedness
        else:
            player_num = 2
            handedness = self.p2_handedness

        # --- 3. Torso midline x ---
        cx_vals = []
        for ji in _CENTRE_JOINTS:
            if ji >= len(hitting_player):
                continue
            jx, jy = hitting_player[ji]
            if jx == 0 and jy == 0:
                continue
            cx_vals.append(float(jx))

        if not cx_vals:
            return 'SHOT'
        player_cx = float(np.mean(cx_vals))

        # --- 4. Overhead shot detection (ball above the player's head) ---
        nose_x, nose_y = hitting_player[_NOSE] if _NOSE < len(hitting_player) else (0, 0) 
        has_nose = not (nose_x == 0 and nose_y == 0) 
        
        # Use shoulders as fallback if nose is missing/unreliable 
        shoulder_ys = [] 
        for ji in [_L_SHOULDER, _R_SHOULDER]: 
            if ji < len(hitting_player): 
                sx, sy = hitting_player[ji] 
                if not (sx == 0 and sy == 0): 
                    shoulder_ys.append(sy) 
                    
        if shoulder_ys: 
            head_y = min(shoulder_ys) 
        else: head_y = nose_y 
        
        if has_nose or shoulder_ys: 
            vertical_dist = head_y - ball_y 
            horizontal_dist = abs(ball_x - player_cx) 
            
            min_vertical_px = 20 # tune this if needed 
            
        # NEW CONDITION: must be truly overhead (not just slightly above + wide) 
        if vertical_dist > min_vertical_px and vertical_dist > 1.2 * horizontal_dist: 
            ankle_ys = [] 
            for ji in _ANKLE_JOINTS: 
                if ji >= len(hitting_player): 
                    continue 
                ax, ay = hitting_player[ji] 
                if ax == 0 and ay == 0: 
                    continue 
                ankle_ys.append(float(ay)) 
            if ankle_ys: 
                ankle_y_mean = float(np.mean(ankle_ys)) 
                if player_num == 1: 
                    at_baseline = ankle_y_mean < frame_height * self.baseline_fraction 
                else: 
                    at_baseline = ankle_y_mean > frame_height * (1.0 - self.baseline_fraction) 
                return 'SERVE' if at_baseline else 'SMASH' 
            else: 
                return 'SMASH'

        # --- 5. Groundstroke: FOREHAND vs BACKHAND ---
        ball_is_left = ball_x < player_cx

        if player_num == 1:
            # Player 1 faces downward (top of screen).
            # Righty: racket arm is on the right side → left-of-body = forehand.
            if handedness == 'right':
                return 'FOREHAND' if ball_is_left else 'BACKHAND'
            else:
                return 'BACKHAND' if ball_is_left else 'FOREHAND'
        else:
            # Player 2 faces upward (bottom of screen) — left/right is mirrored
            # relative to broadcast view vs player's own body frame.
            # Righty: racket arm is on their right, which appears on the LEFT of
            # screen → ball right-of-body on screen = forehand.
            if handedness == 'right':
                return 'FOREHAND' if not ball_is_left else 'BACKHAND'
            else:
                return 'BACKHAND' if not ball_is_left else 'FOREHAND'

    # ------------------------------------------------------------------
    # Signal 1 & 2: Ball trajectory sustained direction reversal
    # ------------------------------------------------------------------

    def _get_trajectory_shot_frames(self, ball_positions: list, axis: str = 'y') -> list[int]:
        """
        Identify frames where the ball's direction along `axis` ('x' or 'y')
        reverses and that reversal is sustained for at least `minimum_change_frames`.

        Adapted from: https://github.com/abdullahtarek/tennis_analysis
        """
        if ball_positions and isinstance(ball_positions[0], dict):
            raw = [x.get(1, []) for x in ball_positions]
            df = pd.DataFrame(raw, columns=['x1', 'y1', 'x2', 'y2'])
        else:
            rows = []
            for pos in ball_positions:
                cx, cy = pos
                if cx is None or cy is None:
                    rows.append([np.nan, np.nan, np.nan, np.nan])
                else:
                    rows.append([cx - 1, cy - 1, cx + 1, cy + 1])
            df = pd.DataFrame(rows, columns=['x1', 'y1', 'x2', 'y2'])

        df['ball_hit'] = 0
        df['mid'] = (df['y1'] + df['y2']) / 2 if axis == 'y' else (df['x1'] + df['x2']) / 2

        df['mid_rolling'] = (
            df['mid']
            .rolling(window=self.rolling_window, min_periods=1, center=False)
            .mean()
        )
        df['delta'] = df['mid_rolling'].diff()

        lookahead = int(self.minimum_change_frames * 1.2)

        for i in range(1, len(df) - lookahead):
            d_i = df['delta'].iloc[i]
            d_next = df['delta'].iloc[i + 1]

            negative_change = d_i > 0 and d_next < 0
            positive_change = d_i < 0 and d_next > 0

            if not (negative_change or positive_change):
                continue

            change_count = 0
            for j in range(i + 1, i + lookahead + 1):
                d_j = df['delta'].iloc[j]
                if negative_change and d_i > 0 and d_j < 0:
                    change_count += 1
                elif positive_change and d_i < 0 and d_j > 0:
                    change_count += 1

            if change_count >= self.minimum_change_frames:
                df.loc[i, 'ball_hit'] = 1

        return df[df['ball_hit'] == 1].index.tolist()

    # ------------------------------------------------------------------
    # Signal 3: Wrist closing velocity (direction-agnostic)
    # ------------------------------------------------------------------

    def _get_closing_velocity_frames(
        self,
        ball_positions: list,
        pose_detections: list,
    ) -> list[int]:
        """
        Find frames where a wrist/elbow reaches its closest approach to the ball
        — i.e. the wrist-ball distance is a local minimum below `closing_velocity_px`.

        The derivative of distance crosses zero from negative (converging) to positive
        (diverging) at the moment of contact, regardless of shot direction.
        """
        n = len(ball_positions)
        min_dists = []

        for frame_idx in range(n):
            pos = ball_positions[frame_idx]
            if isinstance(pos, dict):
                bbox = pos.get(1, [])
                if not bbox or len(bbox) < 4:
                    min_dists.append(np.nan)
                    continue
                bx = (bbox[0] + bbox[2]) / 2
                by = (bbox[1] + bbox[3]) / 2
            else:
                bx, by = pos
                if bx is None or by is None:
                    min_dists.append(np.nan)
                    continue

            kp = pose_detections[frame_idx] if frame_idx < len(pose_detections) else None
            if kp is None:
                min_dists.append(np.nan)
                continue

            best = np.inf
            for player_joints in kp.xy:
                for joint_idx in CONTACT_JOINTS:
                    if joint_idx >= len(player_joints):
                        continue
                    jx, jy = player_joints[joint_idx]
                    if jx == 0 and jy == 0:
                        continue
                    d = distance.euclidean((bx, by), (jx, jy))
                    if d < best:
                        best = d

            min_dists.append(best if best < np.inf else np.nan)

        # Find local minima where distance is below closing_velocity_px
        contact_frames = []
        contact_dists = []
        for i in range(1, n - 1):
            d_prev = min_dists[i - 1]
            d_curr = min_dists[i]
            d_next = min_dists[i + 1]

            if np.isnan(d_prev) or np.isnan(d_curr) or np.isnan(d_next):
                continue

            is_local_min = d_curr < d_prev and d_curr < d_next
            if is_local_min and d_curr <= self.closing_velocity_px:
                contact_frames.append(i)
                contact_dists.append(d_curr)

        return contact_frames, contact_dists

    # ------------------------------------------------------------------
    # Pose proximity gate (shared by all signals)
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

        for player_joints in kp.xy:
            for joint_idx in CONTACT_JOINTS:
                if joint_idx >= len(player_joints):
                    continue
                jx, jy = player_joints[joint_idx]
                if jx == 0 and jy == 0:
                    continue
                if distance.euclidean((bx, by), (jx, jy)) <= self.wrist_proximity_px:
                    return True

        return False