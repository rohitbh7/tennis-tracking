import cv2
import numpy as np
import pandas as pd


# COCO keypoint indices for contact joints (elbows + wrists)
_CONTACT_JOINTS = [7, 8, 9, 10]


class BounceTracker:
    """
    Detects ball bounces on the court using two complementary signals:

    1. **Y-reversal** (primary): the ball falls (delta_y > 0), hits the court, then rises
       (delta_y < 0). The reversal must persist for at least `minimum_change_frames` frames.
       Each raw reversal is snapped to the nearest raw y-maximum in a ±2 frame window so
       the reported frame corresponds to the actual contact point rather than a rolling-
       average lag artifact.

    2. **Speed + delta-x** (secondary): at court contact the ball loses kinetic energy,
       producing a local minimum in frame-to-frame speed sqrt(Δx²+Δy²). Candidates from
       this path require a co-located y-reversal but bypass the stricter sustain check.

    Shot suppression
    ----------------
    When `pose_detections` is provided, suppression is done via **wrist proximity**:
    if any player's elbow or wrist is within `wrist_proximity_px` of the ball at the
    candidate frame, it is classified as player contact (a shot) and suppressed.
    Because a bounce precedes the subsequent shot by several frames, the player's wrist
    is typically 80-200 px from the ball at the bounce frame but ≤80 px at actual contact.

    When pose data is not available, a fallback `shot_suppression_window` (frame-count
    proximity to detected shot frames) is used instead.

    Usage
    -----
        tracker = BounceTracker()
        bounce_frames = tracker.detect_bounces(
            ball_positions,
            shot_frames=shot_frames,
            pose_detections=pose_detections,
        )
        annotated = tracker.draw_bounces(frames, bounce_frames)
    """

    def __init__(
        self,
        minimum_change_frames: int = 7,
        rolling_window: int = 3,
        trajectory_outlier_px: float = 40.0,
        dedup_window: int = 15,
        shot_suppression_window: int = 5,      # frames around known shot to suppress
        wrist_proximity_px: float = 150.0,     # wrist closer than this = player contact
        speed_drop_fraction: float = 0.50,     # speed must drop 50%+ below surroundings
        speed_half_window: int = 4,
        speed_shot_suppression: int = 3,       # window for speed path shot suppression
        peak_descent_px: float = 20.0,        # ball must drop this many px into the local peak
        peak_look_back: int = 8,               # frames to look back for the descent check
        peak_look_forward: int = 2,               # frames ahead to check for continued descent
        peak_forward_tolerance_px: float = 10.0,  # if ball goes this much higher after i, skip
        persist_frames: int = 25,
        ellipse_color: tuple = (0, 200, 255),
        ellipse_axes: tuple = (18, 7),
        ellipse_thickness: int = 2,
    ):
        self.minimum_change_frames = minimum_change_frames
        self.rolling_window = rolling_window
        self.trajectory_outlier_px = trajectory_outlier_px
        self.dedup_window = dedup_window
        self.shot_suppression_window = shot_suppression_window
        self.wrist_proximity_px = wrist_proximity_px
        self.speed_drop_fraction = speed_drop_fraction
        self.speed_half_window = speed_half_window
        self.speed_shot_suppression = speed_shot_suppression
        self.peak_descent_px = peak_descent_px
        self.peak_look_back = peak_look_back
        self.peak_look_forward = peak_look_forward
        self.peak_forward_tolerance_px = peak_forward_tolerance_px
        self.persist_frames = persist_frames
        self.ellipse_color = ellipse_color
        self.ellipse_axes = ellipse_axes
        self.ellipse_thickness = ellipse_thickness

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_bounces(
        self,
        ball_positions: list,
        shot_frames: dict | None = None,
        pose_detections: list | None = None,
    ) -> dict[int, tuple]:
        """
        Parameters
        ----------
        ball_positions  : list of (x, y) tuples or {1: [x1,y1,x2,y2]} dicts,
                          one per frame.
        shot_frames     : optional dict from ShotTracker.detect_shots(). Used as a
                          fallback for shot suppression when pose_detections is None.
        pose_detections : optional list of sv.KeyPoints | None, one per frame.
                          When provided, suppression is done via wrist proximity rather
                          than shot-frame proximity, which correctly handles bounces
                          that immediately precede a shot.

        Returns
        -------
        dict mapping frame_index -> (ball_cx, ball_cy)
        """
        df = self._build_dataframe(ball_positions)
        df = self._remove_outliers(df)

        df['mid_y_rolling'] = (
            df['mid_y']
            .rolling(window=self.rolling_window, min_periods=1, center=False)
            .mean()
        )
        df['delta'] = df['mid_y_rolling'].diff()

        df['speed'] = (df['mid_x'].diff() ** 2 + df['mid_y'].diff() ** 2).pow(0.5)
        df['speed_smooth'] = (
            df['speed']
            .rolling(window=3, min_periods=1, center=True)
            .mean()
        )

        shot_set = set(shot_frames.keys()) if shot_frames else set()

        # ------------------------------------------------------------------
        # Path 1: y-reversal (primary)
        # Snap each raw reversal to the actual raw y-maximum in the vicinity
        # rather than applying a fixed -1 offset, so the reported frame matches
        # the physical contact frame regardless of rolling-average lag variance.
        # ------------------------------------------------------------------
        raw_primary = self._find_reversal_frames(df)
        primary_candidates = sorted(set(
            self._snap_to_y_peak(df, f) for f in raw_primary
        ))
        primary_filtered = self._apply_shot_suppression(
            primary_candidates, shot_set, df, pose_detections,
            window=self.shot_suppression_window,
        )

        # ------------------------------------------------------------------
        # Path 2: speed-drop + y-reversal (secondary)
        # Speed candidates are also snapped to the nearby y-peak so both paths
        # agree on the contact frame.
        # ------------------------------------------------------------------
        raw_speed = self._find_speed_bounce_frames(df)
        speed_candidates = sorted(set(
            self._snap_to_y_peak(df, f) for f in raw_speed
        ))
        speed_filtered = self._apply_shot_suppression(
            speed_candidates, shot_set, df, pose_detections,
            window=self.speed_shot_suppression,
        )

        # ------------------------------------------------------------------
        # Path 3: local y-peak (tertiary)
        # Catches dribble bounces where the ball barely reverses before being hit —
        # the sustained-reversal and speed-drop thresholds both fail for these.
        # Finds raw y-maxima where the ball descended meaningfully into the peak and
        # the peak is near court level (top-60% of observed y values).
        # No snap needed — these are already at the raw y-maximum.
        # ------------------------------------------------------------------
        raw_peak = self._find_local_peak_frames(df)
        peak_filtered = self._apply_shot_suppression(
            raw_peak, shot_set, df, pose_detections,
            window=self.shot_suppression_window,
        )

        merged = sorted(set(primary_filtered) | set(speed_filtered) | set(peak_filtered))

        deduped = []
        for f in merged:
            if deduped and (f - deduped[-1]) < self.dedup_window:
                continue
            deduped.append(f)

        bounce_dict = {}
        for frame_idx in deduped:
            cx = df['mid_x'].iloc[frame_idx]
            cy = df['mid_y'].iloc[frame_idx]
            if np.isnan(cx) or np.isnan(cy):
                continue
            bounce_dict[frame_idx] = (int(cx), int(cy))

        return bounce_dict

    def draw_bounces(self, frames: list, bounce_frames: dict) -> list:
        """
        Draw a fading horizontal oval and 'BOUNCE' label at each bounce location
        for `persist_frames` frames.
        """
        output_frames = []
        for frame_idx, frame in enumerate(frames):
            annotated = frame.copy()
            for bounce_idx, (bx, by) in bounce_frames.items():
                if bounce_idx <= frame_idx < bounce_idx + self.persist_frames:
                    age = frame_idx - bounce_idx
                    alpha = 1.0 - age / self.persist_frames
                    color = tuple(int(c * alpha) for c in self.ellipse_color)
                    cv2.ellipse(
                        annotated,
                        (bx, by),
                        self.ellipse_axes,
                        0,
                        0, 360,
                        color,
                        self.ellipse_thickness,
                        cv2.LINE_AA,
                    )
                    if age < 8:
                        label_alpha = 1.0 - age / 8
                        label_color = tuple(int(c * label_alpha) for c in self.ellipse_color)
                        cv2.putText(
                            annotated,
                            'BOUNCE',
                            (bx + self.ellipse_axes[0] + 4, by + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            label_color,
                            2,
                            cv2.LINE_AA,
                        )
            output_frames.append(annotated)
        return output_frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snap_to_y_peak(self, df: pd.DataFrame, raw_frame: int) -> int:
        """
        Return the frame in [raw_frame-1, raw_frame] with the highest raw
        mid_y value. A window=3 trailing rolling average lags 0-1 frames, so
        a one-frame look-back is sufficient. Larger windows risk snapping into
        the subsequent shot frame and triggering false suppression.
        """
        lo = max(0, raw_frame - 1)
        hi = min(len(df) - 1, raw_frame)
        sub = df['mid_y'].iloc[lo: hi + 1]
        return int(sub.idxmax())

    def _apply_shot_suppression(
        self,
        candidates: list[int],
        shot_set: set,
        df: pd.DataFrame,
        pose_detections: list | None,
        window: int,
    ) -> list[int]:
        """
        Filter bounce candidates that are actually player contacts.

        Primary gate: suppress any candidate within `window` frames of a known
        shot frame (from ShotTracker).

        Rescue clause: if pose_detections is available and NO player's elbow or
        wrist is within wrist_proximity_px of the ball at the candidate frame,
        the candidate is un-suppressed. This handles the bounce-immediately-before-
        shot case: at the bounce frame the wrist is still far from the ball (>150 px);
        at the actual contact frame it is ≤150 px.

        When pose data is unavailable the window-only suppression applies with no rescue.
        """
        filtered = []
        for f in candidates:
            near_shot = any(abs(f - sf) <= window for sf in shot_set)
            if not near_shot:
                filtered.append(f)
                continue
            # Candidate is near a shot frame — rescue if pose confirms no wrist contact
            if pose_detections is not None and not self._wrist_near_ball(df, f, pose_detections):
                filtered.append(f)
            # else: suppress (no pose, or wrist is within proximity → player contact)
        return filtered

    def _wrist_near_ball(
        self,
        df: pd.DataFrame,
        frame_idx: int,
        pose_detections: list,
    ) -> bool:
        """
        Return True if any player's elbow or wrist is within wrist_proximity_px
        of the ball at exactly the given frame.

        At a genuine bounce the player's wrist is typically 150-300 px away.
        At actual racket contact it is ≤150 px (racket extends ~50 px from wrist).
        The 150 px threshold separates these two cases for the rescue clause in
        _apply_shot_suppression.
        """
        if frame_idx >= len(pose_detections):
            return False
        kp = pose_detections[frame_idx]
        if kp is None:
            return False

        bx = df['mid_x'].iloc[frame_idx]
        by = df['mid_y'].iloc[frame_idx]
        if np.isnan(bx) or np.isnan(by):
            return False

        xy = kp.xy
        for player_idx in range(xy.shape[0]):
            for joint_idx in _CONTACT_JOINTS:
                if joint_idx >= xy.shape[1]:
                    continue
                wx, wy = xy[player_idx, joint_idx]
                if wx == 0 and wy == 0:
                    continue
                if np.sqrt((wx - bx) ** 2 + (wy - by) ** 2) <= self.wrist_proximity_px:
                    return True
        return False

    def _build_dataframe(self, ball_positions: list) -> pd.DataFrame:
        if ball_positions and isinstance(ball_positions[0], dict):
            raw = [x.get(1, []) for x in ball_positions]
            df = pd.DataFrame(raw, columns=['x1', 'y1', 'x2', 'y2'])
            df['mid_x'] = (df['x1'] + df['x2']) / 2
            df['mid_y'] = (df['y1'] + df['y2']) / 2
        else:
            rows = []
            for pos in ball_positions:
                cx, cy = pos
                if cx is None or cy is None:
                    rows.append([np.nan, np.nan])
                else:
                    rows.append([float(cx), float(cy)])
            df = pd.DataFrame(rows, columns=['mid_x', 'mid_y'])
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace single-frame spikes with linear interpolation between neighbors.
        A frame is an outlier if its position deviates more than trajectory_outlier_px
        from the midpoint of its two neighbors.
        """
        for col in ['mid_x', 'mid_y']:
            vals = df[col].values.astype(float)
            for i in range(1, len(vals) - 1):
                if np.isnan(vals[i - 1]) or np.isnan(vals[i]) or np.isnan(vals[i + 1]):
                    continue
                expected = (vals[i - 1] + vals[i + 1]) / 2.0
                if abs(vals[i] - expected) > self.trajectory_outlier_px:
                    vals[i] = np.nan
            df[col] = pd.Series(vals, index=df.index).interpolate(method='linear')
        return df

    def _find_local_peak_frames(self, df: pd.DataFrame) -> list[int]:
        """
        Detect bounce candidates as raw y-maxima where the ball has descended
        meaningfully into the peak from a higher position and the peak is near
        court level.

        Unlike _find_reversal_frames this does NOT require the upward reversal to
        be sustained. It catches "dribble" bounces where the ball barely rises
        (or stays flat) before the player contacts it — cases where
        minimum_change_frames cannot be met because the reversal only lasts 1-3
        frames before the shot.

        Local-max criterion (non-strict so the first frame of a flat plateau is
        accepted as the peak):
            mid_y[i] >= mid_y[i-1]  AND  mid_y[i] >= mid_y[i+1]
            AND at least one of those two inequalities is strict
        This picks the first frame of a plateau rather than every frame in it,
        and avoids triggering on monotone ramps.

        Court-proximity filter: the peak must be at or above the 60th percentile
        of all observed y values. Bounce y-values are near the bottom of the
        frame (high y); shot peaks during flight are near the top (low y).

        Look-forward filter (key to avoiding mid-descent plateaus):
        After finding a local max at i, check that no frame in the next
        peak_look_forward frames has a y value more than peak_forward_tolerance_px
        above yi. If the ball continues to a substantially higher y, this is a
        plateau mid-descent, not the true bounce peak. After a real bounce the
        ball rises (y decreases), so this check naturally passes.

        peak_look_forward is kept very short (default 2) for two reasons:
        1. A first-contact plateau (ball lands and rolls flat for 2-3 frames at
           court y before the tracker shows it descending further) should be
           accepted — a 2-frame window catches only the immediate next frames.
        2. Dribble bounces where the ball slowly creeps back to court level over
           6-10 frames before the player contacts it must not be suppressed.
        """
        n = len(df)
        y = df['mid_y'].values

        valid_y = y[~np.isnan(y)]
        if len(valid_y) == 0:
            return []
        court_threshold = np.percentile(valid_y, 60)

        candidates = []
        for i in range(self.peak_look_back, n - 2):
            yi = y[i]
            if np.isnan(yi) or yi < court_threshold:
                continue

            y_prev = y[i - 1]
            y_next = y[i + 1]
            if np.isnan(y_prev) or np.isnan(y_next):
                continue

            # Non-strict local max with at least one strict inequality
            if not (yi >= y_prev and yi >= y_next and (yi > y_prev or yi > y_next)):
                continue

            # Ball must have descended meaningfully into this peak
            pre = y[max(0, i - self.peak_look_back): i]
            pre = pre[~np.isnan(pre)]
            if len(pre) == 0 or yi - np.min(pre) < self.peak_descent_px:
                continue

            # Ball must NOT reach a substantially higher y in the next few frames.
            # If it does, this is a mid-descent plateau, not the true peak.
            # After a real bounce the ball rises (y decreases), so this check
            # naturally passes for genuine contact frames.
            post = y[i + 1: min(n, i + 1 + self.peak_look_forward)]
            post = post[~np.isnan(post)]
            if len(post) > 0 and np.max(post) > yi + self.peak_forward_tolerance_px:
                continue

            candidates.append(i)

        return candidates

    def _find_reversal_frames(self, df: pd.DataFrame) -> list[int]:
        """
        Find frames where delta_y switches from positive (descending) to negative
        (ascending) and the upward direction is sustained for minimum_change_frames.
        """
        lookahead = int(self.minimum_change_frames * 1.2)
        candidates = []

        for i in range(1, len(df) - lookahead):
            d_i = df['delta'].iloc[i]
            d_next = df['delta'].iloc[i + 1]

            if not (d_i > 0 and d_next < 0):
                continue

            d_prev = df['delta'].iloc[i - 1] if i > 0 else np.nan
            if not np.isnan(d_prev) and d_prev <= 0:
                continue

            change_count = sum(
                1 for j in range(i + 1, i + lookahead + 1)
                if df['delta'].iloc[j] < 0
            )
            if change_count >= self.minimum_change_frames:
                candidates.append(i)

        return candidates

    def _find_speed_bounce_frames(self, df: pd.DataFrame) -> list[int]:
        """
        Find bounce candidates using frame-to-frame speed sqrt(Δx² + Δy²).

        At court contact the ball loses kinetic energy — speed collapses to a local
        minimum and then sharply recovers as the ball rebounds. The speed_drop_fraction
        threshold is set high (0.50) to require a genuine 50% collapse, which filters
        out slow gliding and duplicate-frame artifacts (where two identical tracker
        positions create a spurious zero-speed frame without a real ball deceleration).

        A candidate is accepted when:
        - speed_smooth drops 50%+ below the mean of the speed_half_window frames
          on each side (significant speed collapse, not a noise blip)
        - The rolling y-delta changes from positive to negative within ±2 frames
          (confirming a direction reversal co-occurs with the speed drop)
        """
        n = len(df)
        hw = self.speed_half_window
        candidates = []

        for i in range(hw + 1, n - hw - 1):
            s_i = df['speed_smooth'].iloc[i]
            if np.isnan(s_i) or s_i < 1.0:
                continue

            pre_speeds = df['speed_smooth'].iloc[i - hw: i].dropna().tolist()
            post_speeds = df['speed_smooth'].iloc[i + 1: i + hw + 1].dropna().tolist()
            surrounding = pre_speeds + post_speeds
            if not surrounding:
                continue

            mean_surround = np.mean(surrounding)
            if mean_surround <= 0:
                continue

            if s_i > mean_surround * (1.0 - self.speed_drop_fraction):
                continue

            reversal_found = False
            for offset in range(-2, 3):
                j = i + offset
                if j < 1 or j >= n - 1:
                    continue
                if df['delta'].iloc[j] > 0 and df['delta'].iloc[j + 1] < 0:
                    reversal_found = True
                    break

            if reversal_found:
                candidates.append(i)

        return candidates
