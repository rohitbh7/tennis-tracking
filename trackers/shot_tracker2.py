"""
ShotTracker2 — minimal two-signal shot detector.

A shot is registered at frame i when BOTH conditions hold:

  1. Wrist proximity local minimum
     A player's wrist or elbow is closer to the ball at frame i than at
     frame i-1 and frame i+1, AND that minimum distance is within
     wrist_proximity_px.  This means the player's arm was converging on
     the ball and has just started moving away — the physical signature
     of contact, regardless of shot type or direction.

  2. Harsh ball velocity change
     The ball's velocity vector (averaged over vel_window frames before
     and after the contact frame) changes by at least min_velocity_change
     px/frame, and the post-contact speed exceeds min_post_speed.  This
     distinguishes a real hit from the arm merely passing near a
     slow-rolling or stationary ball.

Serve-toss guard
     Before the first shot of the point has been detected, if the ball
     was held nearly stationary for static_run_min or more consecutive
     frames, the contact event is suppressed.  That pattern means the
     player is holding the ball for a serve toss, not hitting it.

Shot classification
     Each detected hit is labelled FOREHAND, BACKHAND, SMASH, or SERVE
     using the same logic as ShotTracker (see _classify_shot).  The only
     intentional difference is a higher baseline_fraction (0.45 vs 0.35)
     which requires the player's ankles to sit further inside their
     baseline zone before a shot is called SERVE rather than SMASH.
"""

import cv2
import numpy as np
from scipy.spatial import distance as scipy_dist

# COCO keypoint indices for wrist / elbow joints (contact detection)
CONTACT_JOINTS = [7, 8, 9, 10]   # left_elbow, right_elbow, left_wrist, right_wrist

# COCO keypoint indices used for shot classification
_NOSE       = 0
_L_SHOULDER = 5
_R_SHOULDER = 6
_L_HIP      = 11
_R_HIP      = 12
_L_ANKLE    = 15
_R_ANKLE    = 16

_CENTRE_JOINTS = [_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP]
_ANKLE_JOINTS  = [_L_ANKLE, _R_ANKLE]


class ShotTracker2:
    def __init__(
        self,
        wrist_proximity_px: float = 100.0,  # wrist must be within this radius
        min_velocity_change: float = 12.0,  # |Δv| in px/frame to call it a shot
        min_post_speed: float = 7.0,        # ball must move at least this fast after contact
        vel_window: int = 2,                # frames averaged for pre/post velocity
        static_run_min: int = 2,            # consecutive near-static frames = held ball
        static_run_px: float = 8.0,         # max displacement per frame when "held"
        static_run_lookback: int = 25,      # how far back to search for a held run
        dedup_window: int = 15,             # minimum frames between consecutive shots
        marker_color: tuple = (0, 255, 255),
        marker_radius: int = 18,
        marker_thickness: int = 3,
        persist_frames: int = 20,
        # --- Handedness ('right' or 'left') ---
        p1_handedness: str = 'right',
        p2_handedness: str = 'right',
        # Fraction of frame height that counts as "at baseline".
        # Higher than ShotTracker's 0.35 — requires ankles to be further inside
        # the baseline zone before a shot is called SERVE rather than SMASH.
        baseline_fraction: float = 0.45,
    ):
        self.wrist_proximity_px  = wrist_proximity_px
        self.min_velocity_change = min_velocity_change
        self.min_post_speed      = min_post_speed
        self.vel_window          = vel_window
        self.static_run_min      = static_run_min
        self.static_run_px       = static_run_px
        self.static_run_lookback = static_run_lookback
        self.dedup_window        = dedup_window
        self.marker_color        = marker_color
        self.marker_radius       = marker_radius
        self.marker_thickness    = marker_thickness
        self.persist_frames      = persist_frames
        self.p1_handedness       = p1_handedness.lower()
        self.p2_handedness       = p2_handedness.lower()
        self.baseline_fraction   = baseline_fraction

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_shots(
        self,
        ball_positions: list,
        pose_detections: list,
        frame_height: int = 720,
        frame_width: int = 1280,
        **kwargs,
    ) -> dict:
        """
        Parameters
        ----------
        ball_positions  : list of (x, y) tuples or {1: [x1,y1,x2,y2]} dicts
        pose_detections : list of sv.KeyPoints | None, one per frame
        frame_height    : pixel height of the video frames (used for classification)
        frame_width     : pixel width  of the video frames (used for classification)

        Returns
        -------
        dict  frame_index -> (ball_cx, ball_cy, shot_label, player_num)
              shot_label is one of: 'FOREHAND', 'BACKHAND', 'SMASH', 'SERVE'
        """
        n       = len(ball_positions)
        centers = self._extract_centers(ball_positions)
        dists   = self._wrist_ball_distances(centers, pose_detections, n)

        candidates = []
        for i in range(1, n - 1):
            d_prev = dists[i - 1]
            d_curr = dists[i]
            d_next = dists[i + 1]

            # --- Condition 1: wrist-ball local minimum within threshold ---
            if np.isnan(d_prev) or np.isnan(d_curr) or np.isnan(d_next):
                continue
            if not (d_curr < d_prev and d_curr < d_next
                    and d_curr <= self.wrist_proximity_px):
                continue

            # --- Condition 2: harsh ball velocity change near the contact ---
            # The wrist minimum lags the actual impact by 1-3 frames (the wrist
            # keeps closing in briefly after the racket strikes the ball), so the
            # velocity jump physically PRECEDES the wrist minimum.  We search up
            # to search_radius frames BACK, and allow up to i+2 frames FORWARD.
            search_radius = self.vel_window + 1
            found_change = False
            for j in range(max(1, i - search_radius), min(n - 1, i + 3)):
                pre_v  = self._avg_vel(centers, j, forward=False)
                post_v = self._avg_vel(centers, j, forward=True)
                if pre_v is None or post_v is None:
                    continue
                delta_v    = np.hypot(post_v[0] - pre_v[0], post_v[1] - pre_v[1])
                post_speed = np.hypot(post_v[0], post_v[1])
                if delta_v >= self.min_velocity_change and post_speed >= self.min_post_speed:
                    found_change = True
                    break
            if not found_change:
                continue

            candidates.append(i)

        # --- First pass: toss guard + report frame ---
        confirmed = []
        for i in candidates:
            bx, by = centers[i]
            if bx is None:
                continue

            if not confirmed and self._ball_was_held(centers, i):
                continue

            confirmed.append((i, int(bx), int(by), dists[i]))

        # --- Second pass: cluster-dedup (keep closest wrist) ---
        shot_frames: dict = {}
        pending = None   # (d_curr, report_frame, rx, ry)

        for report_frame, rx, ry, d_curr in confirmed:
            if pending is None:
                pending = (d_curr, report_frame, rx, ry)
            elif report_frame - pending[1] < self.dedup_window:
                if d_curr < pending[0]:
                    pending = (d_curr, report_frame, rx, ry)
            else:
                _, pf, px, py = pending
                label, player_num = self._classify_shot(
                    pf, px, py, pose_detections, frame_height, frame_width,
                )
                shot_frames[pf] = (px, py, label, player_num)
                pending = (d_curr, report_frame, rx, ry)

        if pending is not None:
            _, pf, px, py = pending
            label, player_num = self._classify_shot(
                pf, px, py, pose_detections, frame_height, frame_width,
            )
            shot_frames[pf] = (px, py, label, player_num)

        return shot_frames

    def draw_shot_markers(self, frames: list, shot_frames: dict) -> list:
        """Draw a fading circle and shot-type label at each detected contact."""
        _label_colors = {
            'FOREHAND': (0, 255, 100),
            'BACKHAND': (100, 180, 255),
            'SMASH':    (0, 80, 255),
            'SERVE':    (255, 200, 0),
            'SHOT':     (0, 255, 255),   # fallback
        }

        output_frames = []
        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()
            for shot_idx, shot_data in shot_frames.items():
                bx, by   = shot_data[0], shot_data[1]
                label    = shot_data[2] if len(shot_data) > 2 else 'SHOT'
                base_color = _label_colors.get(label, self.marker_color)

                if shot_idx <= frame_idx < shot_idx + self.persist_frames:
                    age   = frame_idx - shot_idx
                    alpha = 1.0 - age / self.persist_frames
                    color = tuple(int(c * alpha) for c in base_color)

                    cv2.circle(
                        annotated, (bx, by),
                        self.marker_radius, color, self.marker_thickness,
                    )
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
    ) -> tuple:
        """
        Return (label, player_num) where label is one of
        'FOREHAND', 'BACKHAND', 'SMASH', or 'SERVE'.

        Steps
        -----
        1. Find the hitter: whichever player's wrist/elbow is closest to the ball.
        2. Determine player number (P1 = top half, P2 = bottom half) and handedness.
        3. Compute torso midline x from shoulders + hips.
        4. Overhead check: if the ball is well above the player's head and more
           vertical than horizontal from the torso midline:
             - ankles near their own baseline  -> SERVE
             - otherwise                       -> SMASH
        5. Groundstroke: FOREHAND or BACKHAND based on ball side x handedness.
        """
        kp = pose_detections[frame_idx] if frame_idx < len(pose_detections) else None
        if kp is None or len(kp.xy) == 0:
            return 'SHOT', None

        # --- 1. Find the hitter by contact-joint proximity to the ball ---
        best_dist      = np.inf
        hitting_player = None
        hitting_avg_y  = None

        for player_joints in kp.xy:
            for ji in CONTACT_JOINTS:
                if ji >= len(player_joints):
                    continue
                jx, jy = player_joints[ji]
                if jx == 0 and jy == 0:
                    continue
                d = scipy_dist.euclidean((ball_x, ball_y), (jx, jy))
                if d < best_dist:
                    best_dist      = d
                    hitting_player = player_joints
                    valid_ys = [
                        player_joints[j][1]
                        for j in range(len(player_joints))
                        if not (player_joints[j][0] == 0 and player_joints[j][1] == 0)
                    ]
                    hitting_avg_y = float(np.mean(valid_ys)) if valid_ys else None

        if hitting_player is None or hitting_avg_y is None:
            return 'SHOT', None

        # --- 2. Player number and handedness ---
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
            return 'SHOT', player_num
        player_cx = float(np.mean(cx_vals))

        # --- 4. Overhead shot detection ---
        nose_x, nose_y = hitting_player[_NOSE] if _NOSE < len(hitting_player) else (0, 0)
        has_nose = not (nose_x == 0 and nose_y == 0)

        shoulder_ys = []
        for ji in [_L_SHOULDER, _R_SHOULDER]:
            if ji < len(hitting_player):
                sx, sy = hitting_player[ji]
                if not (sx == 0 and sy == 0):
                    shoulder_ys.append(sy)

        if shoulder_ys:
            head_y = min(shoulder_ys)
        else:
            head_y = nose_y

        if has_nose or shoulder_ys:
            vertical_dist   = head_y - ball_y
            horizontal_dist = abs(ball_x - player_cx)
            min_vertical_px = 20

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
                return ('SERVE' if at_baseline else 'SMASH'), player_num
            else:
                return 'SMASH', player_num

        # --- 5. Groundstroke: FOREHAND vs BACKHAND ---
        ball_is_left = ball_x < player_cx

        if player_num == 1:
            if handedness == 'right':
                return ('FOREHAND' if ball_is_left else 'BACKHAND'), player_num
            else:
                return ('BACKHAND' if ball_is_left else 'FOREHAND'), player_num
        else:
            if handedness == 'right':
                return ('FOREHAND' if not ball_is_left else 'BACKHAND'), player_num
            else:
                return ('BACKHAND' if not ball_is_left else 'FOREHAND'), player_num

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_centers(self, ball_positions: list) -> list:
        """Convert ball_positions to a list of (x, y) or (None, None)."""
        centers = []
        for pos in ball_positions:
            if isinstance(pos, dict):
                bbox = pos.get(1, [])
                if bbox and len(bbox) >= 4:
                    centers.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
                else:
                    centers.append((None, None))
            else:
                bx, by = pos
                centers.append((bx, by) if bx is not None else (None, None))
        return centers

    def _wrist_ball_distances(
        self, centers: list, pose_detections: list, n: int
    ) -> list:
        """Minimum wrist/elbow-ball distance for every frame (NaN when unavailable)."""
        dists = []
        for i in range(n):
            bx, by = centers[i]
            if bx is None:
                dists.append(float('nan'))
                continue
            kp = pose_detections[i] if i < len(pose_detections) else None
            if kp is None:
                dists.append(float('nan'))
                continue
            best = float('inf')
            for player_joints in kp.xy:
                for ji in CONTACT_JOINTS:
                    if ji >= len(player_joints):
                        continue
                    jx, jy = player_joints[ji]
                    if jx == 0 and jy == 0:
                        continue
                    d = scipy_dist.euclidean((bx, by), (jx, jy))
                    if d < best:
                        best = d
            dists.append(best if best < float('inf') else float('nan'))
        return dists

    def _avg_vel(self, centers: list, frame_idx: int, forward: bool):
        """
        Average velocity vector over vel_window steps before (forward=False)
        or after (forward=True) frame_idx.  Returns (vx, vy) or None.

        Exact-duplicate consecutive positions (same x AND same y) are skipped:
        they are stale ball-tracker detections, not the ball genuinely standing
        still, and including them drags the velocity estimate toward zero.
        """
        n   = len(centers)
        vxs = []
        vys = []
        if forward:
            for k in range(frame_idx, min(frame_idx + self.vel_window, n - 1)):
                ax, ay = centers[k + 1]
                bx, by = centers[k]
                if ax is None or bx is None:
                    continue
                if ax == bx and ay == by:
                    continue
                vxs.append(ax - bx)
                vys.append(ay - by)
        else:
            for k in range(frame_idx, max(frame_idx - self.vel_window, 0), -1):
                ax, ay = centers[k]
                bx, by = centers[k - 1]
                if ax is None or bx is None:
                    continue
                if ax == bx and ay == by:
                    continue
                vxs.append(ax - bx)
                vys.append(ay - by)
        if not vxs:
            return None
        return float(np.mean(vxs)), float(np.mean(vys))

    def _ball_was_held(self, centers: list, frame_idx: int) -> bool:
        """
        Return True if the ball was nearly stationary for >= static_run_min
        consecutive frames in the window ending at frame_idx AND that static
        run was not preceded by significant ball movement.

        The second condition distinguishes a player genuinely holding the ball
        (serve preparation) from a toss apex (ball momentarily stops at the top
        of the arc but was already moving before that).
        """
        start         = max(0, frame_idx - self.static_run_lookback)
        available     = frame_idx - start
        effective_min = max(2, min(self.static_run_min, available))
        if available < effective_min:
            return False

        run           = 0
        run_start     = None
        best_run      = 0
        best_run_start = None

        for k in range(start + 1, frame_idx + 1):
            a, b = centers[k - 1], centers[k]
            if a[0] is None or b[0] is None:
                run = 0; run_start = None
                continue
            if np.hypot(a[0] - b[0], a[1] - b[1]) <= self.static_run_px:
                if run == 0:
                    run_start = k - 1
                run += 1
                if run > best_run:
                    best_run = run
                    best_run_start = run_start
            else:
                run = 0; run_start = None

        if best_run < effective_min:
            return False

        if best_run_start <= start + effective_min:
            return True  # run starts at the beginning of lookback — no prior history, assume held

        pre_move_px = self.static_run_px
        for m in range(max(0, best_run_start - 5), best_run_start):
            if m + 1 >= len(centers):
                continue
            ma, mb = centers[m], centers[m + 1]
            if ma[0] is None or mb[0] is None:
                continue
            if np.hypot(ma[0] - mb[0], ma[1] - mb[1]) > pre_move_px:
                return False
        return True
