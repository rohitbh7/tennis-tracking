import cv2
import numpy as np
import pandas as pd


# COCO keypoint indices for contact joints (elbows + wrists)
_CONTACT_JOINTS = [7, 8, 9, 10]


class BounceTracker:
    """
    Detects ball bounces on the court using four complementary detection paths:

    1. **Y-reversal** (primary): the ball falls (delta_y > 0), hits the court, then rises
       (delta_y < 0). The reversal must persist for at least `minimum_change_frames` frames.

    2. **Speed-drop + y-reversal** (secondary): at court contact the ball loses kinetic
       energy, producing a local speed minimum co-located with a y-reversal.

    3. **Local y-peak** (tertiary): raw y-maxima where the ball descended meaningfully
       into the peak and the peak is near court level. Catches dribble/short bounces.
       Uses a STRICT local-max criterion (both neighbours strictly below yi) to avoid
       firing on mid-descent plateaus.

    4. **Descent arrival** (quaternary): detects half-volleys — bounces immediately
       followed by a shot. Requires a clear falling arc into court level (50th pct
       threshold) with at least `descent_arrival_min_falling` consecutive descending
       frames. Does not require a sustained post-bounce rise since the player's shot
       may intervene immediately.

    X-flip suppression
    ------------------
    At a genuine bounce the ball continues in the same horizontal direction (friction
    slows it but does not reverse it). When a player hits the ball, x-direction reverses.

    The check uses an ASYMMETRIC window: long look-back (5 frames, confirms approach
    direction) and short look-forward (x_flip_post_window, default 2 frames). This is
    critical for half-volleys — by the time a shot sends the ball back, the ball has
    already left the bounce frame, so a 2-frame post-window sees no reversal at the
    bounce frame itself. A long post-window would capture the subsequent shot.

    The magnitude threshold is RELATIVE to the video's y_range (x_flip_rel_mag, default
    3%). Fixed pixel thresholds break when camera zoom changes; a relative threshold
    scales automatically across different videos and camera angles.

    A DUAL x-flip check is applied: both the raw detection frame and the snapped frame
    must pass. This catches descent-phase detections that snap forward into a shot frame
    where the reversal is large and clear.

    Shot suppression
    ----------------
    When `pose_detections` is provided, suppression is done via **wrist proximity**:
    if any player's elbow or wrist is within `wrist_proximity_px` of the ball at the
    candidate frame, it is classified as player contact and suppressed.

    When pose data is not available, a fallback `shot_suppression_window` is used.

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
        shot_suppression_window: int = 5,
        wrist_proximity_px: float = 80.0,
        speed_drop_fraction: float = 0.50,
        speed_half_window: int = 4,
        speed_shot_suppression: int = 3,
        peak_descent_px: float = 20.0,
        peak_look_back: int = 8,
        peak_look_forward: int = 5,
        peak_forward_tolerance_px: float = 30.0,
        # X-flip suppression parameters
        x_flip_post_window: int = 2,        # frames ahead to check for x-reversal
        x_flip_rel_mag: float = 0.03,       # min reversal as fraction of y_range (3%)
        x_flip_pre_window: int = 5,         # frames back to confirm approach direction
        # Descent-arrival (Path 4 / half-volley) parameters
        descent_arrival_back: int = 6,      # lookback window for falling-frame count
        descent_arrival_min_falling: int = 4,  # min frames falling before peak
        descent_arrival_fwd: int = 5,       # frames forward to confirm rise after peak
        # Y-velocity deceleration (Path 5 / low-angle bounce) parameters
        decel_back: int = 5,                # frames back to measure pre-bounce dy
        decel_fwd: int = 5,                 # frames forward to measure post-bounce dy
        decel_ratio: float = 0.40,          # post/pre dy must be < this to qualify
        decel_min_speed_frac: float = 0.04, # min pre-bounce speed as fraction of y_range
        # Speed-ratio filter: suppress when post-contact speed exceeds pre-contact
        # speed by this multiplier. A court bounce dissipates energy (post ≤ pre);
        # a player shot injects energy (post >> pre). Applied globally after x-flip.
        # Set to 4.0 (not 3.0) to accommodate far-court bounces: perspective
        # projection makes the ball appear to accelerate as it rises back from the
        # far baseline, inflating the image-space post/pre ratio to ~3.1.
        speed_shot_multiplier: float = 4.0,
        # Far-court bounce (Path 6) parameters
        far_court_pct: float = 50.0,           # percentile ceiling for far-court y zone
        far_court_approach_back: int = 20,     # frames back to find pre-bounce y max
        far_court_min_approach_px: float = 100.0,  # ball must have come from this far above
        far_court_fwd: int = 8,                # frames forward for post-drop confirmation
        far_court_min_post_drop_px: float = 50.0,  # y must drop this far below bounce point
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
        self.x_flip_post_window = x_flip_post_window
        self.x_flip_rel_mag = x_flip_rel_mag
        self.x_flip_pre_window = x_flip_pre_window
        self.descent_arrival_back = descent_arrival_back
        self.descent_arrival_min_falling = descent_arrival_min_falling
        self.descent_arrival_fwd = descent_arrival_fwd
        self.decel_back = decel_back
        self.decel_fwd = decel_fwd
        self.decel_ratio = decel_ratio
        self.decel_min_speed_frac = decel_min_speed_frac
        self.speed_shot_multiplier = speed_shot_multiplier
        self.far_court_pct = far_court_pct
        self.far_court_approach_back = far_court_approach_back
        self.far_court_min_approach_px = far_court_min_approach_px
        self.far_court_fwd = far_court_fwd
        self.far_court_min_post_drop_px = far_court_min_post_drop_px
        self.persist_frames = persist_frames
        self.ellipse_color = ellipse_color
        self.ellipse_axes = ellipse_axes
        self.ellipse_thickness = ellipse_thickness
        # Set during detect_bounces; used by x-flip helpers
        self._y_range: float = 1.0

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
        shot_frames     : optional dict from ShotTracker.detect_shots().
        pose_detections : optional list of sv.KeyPoints | None, one per frame.

        Returns
        -------
        dict mapping frame_index -> (ball_cx, ball_cy)
        """
        df = self._build_dataframe(ball_positions)
        df = self._remove_outliers(df)
        df = self._remove_stale_positions(df)

        # Store y_range for relative x-flip threshold scaling
        valid_y = df['mid_y'].dropna().values
        self._y_range = float(np.max(valid_y) - np.min(valid_y)) if len(valid_y) > 1 else 1.0

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
        # ------------------------------------------------------------------
        raw_peak = self._find_local_peak_frames(df)
        peak_filtered = self._apply_shot_suppression(
            raw_peak, shot_set, df, pose_detections,
            window=self.shot_suppression_window,
        )

        # ------------------------------------------------------------------
        # Path 4: descent arrival (quaternary — half-volley / quick bounce)
        # Snap with radius=2: moves a pre-peak local-max wobble (e.g. frame
        # 168 at y=316) to the true y-maximum (frame 170 at y=328).  Radius
        # is kept small (vs Path 1's radius=1) to avoid crossing into shot
        # frames; at half-volley peaks the snap stays at the same frame.
        #
        # Shot suppression is intentionally SKIPPED for Path 4.
        # Half-volley bounces — the only class Path 4 detects — always occur
        # simultaneously with a shot: the ball bounces and is struck in the
        # same frame.  The ShotTracker will flag that frame as a shot, and
        # wrist proximity will confirm contact.  Applying shot suppression
        # here would always eliminate the very bounces this path exists to
        # catch.  The global x-flip check below is the sole gate: it uses an
        # asymmetric 5-frame pre / 2-frame post window with a relative
        # magnitude threshold so that a gentle half-volley deflection (< 3%
        # of y_range in 2 frames) is kept while a hard player-redirected shot
        # (large x reversal) is correctly suppressed.
        # ------------------------------------------------------------------
        raw_descent = self._find_descent_arrival_frames(df)
        descent_filtered = sorted(set(
            self._snap_to_y_peak(df, f, radius=2) for f in raw_descent
        ))

        # ------------------------------------------------------------------
        # Path 5: y-velocity deceleration (quinary — low-angle / skidding bounce)
        # Catches bounces where the ball hits the court and continues moving
        # horizontally with dramatically reduced vertical speed. Y never
        # reverses — the ball skids along the surface or bounces at such a
        # shallow angle that the rise is invisible in frame coordinates.
        #
        # The signal is purely kinematic: the mean dy over the preceding
        # decel_back frames drops to less than decel_ratio of itself in the
        # following decel_fwd frames.  No y-reversal, no local y-max, and
        # no snap are required or applied.
        #
        # Shot suppression is skipped for the same reason as Path 4: these
        # bounces often occur just before a shot and applying suppression
        # would eliminate the detection.  The global x-flip check is the
        # sole gate.
        # ------------------------------------------------------------------
        decel_frames = self._find_deceleration_frames(df)

        # ------------------------------------------------------------------
        # Path 6: Far-court bounce (inverse geometry)
        # At the far end of the court (player 1's side), the ball approaches
        # with DECREASING y in image coordinates and the bounce appears as a
        # strict local y-MAXIMUM at low y values, after which y continues to
        # decrease as the ball rises past the far baseline.  All other paths
        # miss this because they expect court contact to coincide with high y
        # (near-court geometry) or a y-reversal from increasing to decreasing.
        # Shot suppression is applied normally; the global x-flip and speed-
        # ratio filters also apply.
        # ------------------------------------------------------------------
        raw_far_court = self._find_far_court_bounce_frames(df)
        far_court_filtered = self._apply_shot_suppression(
            raw_far_court, shot_set, df, pose_detections,
            window=self.shot_suppression_window,
        )

        merged = sorted(
            set(primary_filtered)
            | set(speed_filtered)
            | set(peak_filtered)
            | set(descent_filtered)
            | set(decel_frames)
            | set(far_court_filtered)
        )


        # ------------------------------------------------------------------
        # X-flip suppression (global post-filter)
        # Applied AFTER shot suppression so the two mechanisms are independent.
        # Dual check: suppress if x reverses at the candidate frame itself OR
        # at the snap-snapped frame (catches descent detections that were
        # moved forward into a shot frame by the snap function).
        # ------------------------------------------------------------------
        x_flip_passed = [
            f for f in merged
            if not self._x_flip_suppressed(df, f)
        ]

        # ------------------------------------------------------------------
        # Speed-ratio filter (global post-filter)
        # A genuine court bounce dissipates kinetic energy — the ball leaves
        # the court at similar or lower speed than it arrived.  A player
        # shot injects energy — the ball exits dramatically faster than it
        # arrived.  Suppress any candidate where the mean speed in the next
        # speed_half_window frames exceeds the mean pre-candidate speed by
        # more than speed_shot_multiplier.
        #
        # This catches false positives that survive x-flip (because the ball
        # barely moved horizontally) and survive wrist-proximity rescue (because
        # the pose detector places the wrist just outside the threshold at the
        # exact candidate frame): the tell-tale sign of a player contact is
        # always a sudden large speed increase, regardless of x-direction.
        # ------------------------------------------------------------------
        n_frames = len(df)
        hw = self.speed_half_window
        speed_ratio_passed = []
        for f in x_flip_passed:
            pre_spd = df['speed_smooth'].iloc[max(0, f - hw): f].dropna()
            post_spd = df['speed_smooth'].iloc[f + 1: min(n_frames, f + hw + 1)].dropna()
            if (len(pre_spd) >= 2
                    and len(post_spd) >= 2
                    and float(pre_spd.mean()) > 1.0
                    and float(post_spd.mean()) > float(pre_spd.mean()) * self.speed_shot_multiplier):
                continue   # post-contact speed explosion → player shot, not a bounce
            speed_ratio_passed.append(f)

        deduped = []
        for f in speed_ratio_passed:
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

    def _x_flip_suppressed(self, df: pd.DataFrame, frame: int) -> bool:
        """
        Return True if the ball's x-direction reverses at `frame` by an
        amount large enough to indicate a player shot rather than a bounce.

        Asymmetric windows
        ------------------
        x_flip_pre_window (default 5): long look-back confirms the ball was
          travelling consistently in one direction before the candidate.
        x_flip_post_window (default 2): short look-forward checks only the
          immediate aftermath. Critical for half-volleys: when a bounce is
          immediately followed by a shot, the x-reversal from the shot appears
          2-3 frames later. A short post-window sees the ball's own momentum
          right after contact — still in the original direction — before the
          player can redirect it.

        Relative magnitude threshold
        ----------------------------
        min_mag = y_range * x_flip_rel_mag (default 3% of y_range).
        Fixed pixel thresholds break when camera zoom changes. A relative
        threshold scales automatically:
          - Half-volley pick-up: ball barely deflects in x in 2 frames (<3%)
            -> not suppressed (bounce kept)
          - Hard groundstroke: ball flies far in new direction in 2 frames (>3%)
            -> suppressed (shot removed)

        Dual check
        ----------
        The caller should also check the snapped frame so that descent-phase
        detections snapped into a shot frame are caught. This method handles
        one frame at a time; the caller applies it to both raw and snapped.
        """
        n = len(df)
        x = df['mid_x'].values
        min_mag = self._y_range * self.x_flip_rel_mag

        dx_pre  = x[frame] - x[max(0, frame - self.x_flip_pre_window)]
        dx_post = x[min(n - 1, frame + self.x_flip_post_window)] - x[frame]

        if abs(dx_pre) < min_mag or abs(dx_post) < min_mag:
            return False   # too small to determine direction confidently
        return bool(dx_pre * dx_post < 0)

    def _snap_to_y_peak(self, df: pd.DataFrame, raw_frame: int, radius: int = 1) -> int:
        """
        Return the frame in [raw_frame-radius, raw_frame+radius] with the
        highest raw mid_y value.

        Default radius=1 (Paths 1 & 2): a window=3 trailing rolling average
        lags 0-1 frames, so a small window is sufficient. Keeps the snap tight
        to avoid crossing into adjacent shot frames.

        radius=2 (Path 4 / descent arrival): a pre-peak wobble can place the
        strict local max 1-2 frames before the true bounce peak (e.g., frame
        168 at y=316 before the true peak at frame 170 at y=328). The extra
        reach finds the correct maximum while staying well within the dedup
        window and well away from the subsequent shot frame.

        The dual x-flip check applied after snapping catches any case where a
        larger radius accidentally snaps into a shot frame.
        """
        lo = max(0, raw_frame - radius)
        hi = min(len(df) - 1, raw_frame + radius)
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
        at the actual contact frame it is <= 150 px.
        """
        filtered = []
        for f in candidates:
            near_shot = any(abs(f - sf) <= window for sf in shot_set)
            if not near_shot:
                filtered.append(f)
                continue
            if pose_detections is not None and not self._wrist_near_ball(df, f, pose_detections):
                filtered.append(f)
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

    def _remove_stale_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace frames where the ball position is identical to the previous
        frame with linearly interpolated values.

        Consecutive identical positions are stale/carried-forward detections
        that produce a spurious zero-delta.  That zero contaminates the
        rolling-mean delta used by every detection path — most critically
        Path 5 (deceleration), where a single stale frame in the pre-bounce
        window can drag the pre-mean just below the minimum-speed gate and
        cause a real bounce to be missed.

        After NaN-ing the stale frame, pandas linear interpolation fills it
        with the position midway between its neighbours, giving a delta that
        correctly reflects the ball's actual velocity in that window.
        """
        x = df['mid_x'].values.astype(float).copy()
        y = df['mid_y'].values.astype(float).copy()
        for i in range(1, len(x)):
            if np.isnan(x[i]) or np.isnan(x[i - 1]):
                continue
            if x[i] == x[i - 1] and y[i] == y[i - 1]:
                x[i] = np.nan
                y[i] = np.nan
        df['mid_x'] = pd.Series(x, index=df.index).interpolate(method='linear')
        df['mid_y'] = pd.Series(y, index=df.index).interpolate(method='linear')
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
        Detect bounce candidates as raw y-maxima near court level where the
        ball descended meaningfully into the peak.

        Local-max criterion: STRICT (both neighbours strictly below yi).
        The previous non-strict criterion (yi >= neighbours) fired on flat
        plateaus mid-descent (e.g. tracker noise at y=700 on the way to 772),
        generating false votes that snapped into the subsequent shot frame.
        Strict inequality naturally rejects these plateaus while still
        accepting genuine bounce peaks.

        Court-proximity filter: 60th percentile of observed y values.

        Look-forward filter: the ball must not continue to a substantially
        higher y in the next peak_look_forward frames (would indicate a
        mid-descent plateau, not the true peak).
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

            # Strict local max: both neighbours must be strictly below yi.
            # Rejects mid-descent plateaus that the old non-strict check missed.
            if not (yi > y_prev and yi > y_next):
                continue

            # Ball must have descended meaningfully into this peak
            pre = y[max(0, i - self.peak_look_back): i]
            pre = pre[~np.isnan(pre)]
            if len(pre) == 0 or yi - np.min(pre) < self.peak_descent_px:
                continue

            # Ball must NOT continue to a substantially higher y soon after
            post = y[i + 1: min(n, i + 1 + self.peak_look_forward)]
            post = post[~np.isnan(post)]
            if len(post) > 0 and np.max(post) > yi + self.peak_forward_tolerance_px:
                continue

            candidates.append(i)

        return candidates

    def _find_descent_arrival_frames(self, df: pd.DataFrame) -> list[int]:
        """
        Detect half-volley bounces: the ball lands at court level and is
        immediately struck by the player.

        In these cases the x-direction reversal caused by the shot appears
        at the same frame as the bounce, making x-flip suppression and the
        sustained-reversal threshold both fail. This path detects the bounce
        purely from the approach arc:

        1. The ball is at a STRICT local y-maximum (both neighbours strictly
           below — rejects mid-descent plateaus).
        2. The peak is at or above the 50th percentile of observed y values
           (lower threshold than Path 3 to catch low-altitude bounces near
           the service line that barely exceed the 60th pct cutoff).
        3. At least descent_arrival_min_falling of the preceding
           descent_arrival_back frames have positive delta_y (ball was
           genuinely descending, not just drifting).
        4. The ball dropped at least x_flip_rel_mag * y_range pixels before
           the peak (confirms a meaningful arc, not tracker noise).
        5. At least 1 frame of post-peak rise (y decreases after the peak),
           confirming this is the actual contact point and not a flat section
           mid-descent.

        X-flip suppression is NOT applied internally — the global dual check
        in detect_bounces handles it with the relative magnitude threshold,
        which correctly lets small half-volley deflections through while
        catching hard player-redirected shots.
        """
        n = len(df)
        y = df['mid_y'].values
        dy = df['delta'].values

        valid_y = y[~np.isnan(y)]
        if len(valid_y) == 0:
            return []

        court_threshold_50 = float(np.percentile(valid_y, 50))
        min_descent = self._y_range * self.x_flip_rel_mag  # same relative scale as x-flip

        candidates = []
        back = self.descent_arrival_back
        fwd  = self.descent_arrival_fwd

        for i in range(back, n - fwd):
            yi = y[i]
            if np.isnan(yi) or yi < court_threshold_50:
                continue

            y_prev, y_next = y[i - 1], y[i + 1]
            if np.isnan(y_prev) or np.isnan(y_next):
                continue

            # Strict local max
            if not (yi > y_prev and yi > y_next):
                continue

            # Must have been consistently falling before the peak
            falling = sum(
                1 for j in range(i - back, i)
                if not np.isnan(dy[j]) and dy[j] > 0
            )
            if falling < self.descent_arrival_min_falling:
                continue

            # Meaningful drop magnitude (scale-free)
            pre = y[max(0, i - back): i]
            pre = pre[~np.isnan(pre)]
            if len(pre) == 0 or yi - np.min(pre) < min_descent:
                continue

            # At least 1 frame of post-peak rise confirms this is the true peak
            post = y[i + 1: min(n, i + fwd + 1)]
            post = post[~np.isnan(post)]
            if len(post) == 0 or np.min(post) >= yi:
                continue

            # Forward-tolerance: if any post-peak frame reaches substantially
            # HIGHER y (deeper into the court) than the candidate peak, this
            # is a mid-descent wobble, not the true bounce peak.  A 1-frame
            # dip followed by continued descent (e.g. ball at y=406 with
            # y=484 two frames later) would otherwise look like a brief rise.
            #
            # peak_forward_tolerance_px (default 30) is tight enough to reject
            # mid-flight noise (delta >> 30 px over 5 frames) while allowing
            # genuine rebound trajectories where post-peak y stays near the
            # bounce level.
            if np.max(post) > yi + self.peak_forward_tolerance_px:
                continue

            candidates.append(i)

        return candidates

    def _find_far_court_bounce_frames(self, df: pd.DataFrame) -> list[int]:
        """
        Detect far-court bounces where the ball travels from the near court
        (high y in image coordinates) to the far court (low y) and bounces.

        In image coordinates the far court occupies LOW y values (player 1's
        baseline ≈ y=324, far service line ≈ y=400).  A ball crossing the
        full court appears as y DECREASING on approach, a brief strict local
        y-MAXIMUM at the far end, and y CONTINUING to decrease after contact
        as the ball rises past the far baseline.

        All other paths miss this because:
        • Paths 1 & 2 need delta to switch from positive to negative (y was
          increasing then drops), but here delta stays negative throughout.
        • Paths 3, 4 & 5 require y to be above the 50th / 60th percentile of
          all observed y — far-court y values are in the lower percentiles.

        Criteria
        --------
        1. y[i] ≤ far_court_threshold (50th percentile of all ball y values)
        2. Strict local y-maximum in RAW y: y[i] > y[i-1] AND y[i] > y[i+1].
           Raw rather than rolling-mean values: the bump can be as small as
           3 px and the rolling average would wash it out entirely.
        3. Ball approached from significantly higher y (came from near court):
           max(y[i - back : i]) - y[i] > far_court_min_approach_px.
           This is the strongest discriminator — it rules out any event where
           the ball was already at low y in the lookback window.
        4. After the bounce y must continue to drop meaningfully, confirming
           the ball rose past the far-court surface into the "above baseline"
           image region:
           min(y[i+1 : i+fwd]) < y[i] - far_court_min_post_drop_px.
        """
        n = len(df)
        y = df['mid_y'].values   # processed (stale-removed, outlier-removed)

        valid_y = y[~np.isnan(y)]
        if len(valid_y) == 0:
            return []

        far_court_threshold = float(np.percentile(valid_y, self.far_court_pct))

        back = self.far_court_approach_back
        fwd  = self.far_court_fwd

        candidates = []
        for i in range(back, n - fwd):
            yi = y[i]
            if np.isnan(yi) or yi > far_court_threshold:
                continue

            y_prev = y[i - 1]
            y_next = y[i + 1]
            if np.isnan(y_prev) or np.isnan(y_next):
                continue

            # Strict local y-maximum in raw coordinates
            if not (yi > y_prev and yi > y_next):
                continue

            # Ball must have approached from substantially higher y
            pre = y[i - back: i]
            pre = pre[~np.isnan(pre)]
            if len(pre) == 0:
                continue
            if float(np.max(pre)) - yi < self.far_court_min_approach_px:
                continue

            # y must continue dropping after the bounce — ball rose past the
            # far baseline, projecting to ever-smaller y in image space
            post = y[i + 1: i + fwd + 1]
            post = post[~np.isnan(post)]
            if len(post) == 0:
                continue
            if float(np.min(post)) >= yi - self.far_court_min_post_drop_px:
                continue

            candidates.append(i)

        return candidates

    def _find_deceleration_frames(self, df: pd.DataFrame) -> list[int]:
        """
        Detect low-angle / skidding bounces by finding frames where the
        ball's vertical speed (dy) drops sharply without reversing.

        In a standard bounce, y reverses — the ball rises after contact.
        In a skidding or very shallow-angle bounce, the ball stays low and
        keeps moving forward but loses most of its vertical momentum at
        contact.  Y never decreases; it just stops increasing as fast.

        Detection criterion
        -------------------
        At candidate frame i:
          - Ball is at or below court level (y > 50th pct of observed y)
          - mean(dy[i-back : i])   >  decel_min_speed_frac * y_range
            (ball was meaningfully falling, not just drifting)
          - mean(dy[i : i+fwd])    <  decel_ratio * mean(dy[i-back : i])
            (post-contact dy is less than decel_ratio of pre-contact dy)

        The ratio threshold (default 0.40) accepts only pronounced
        decelerations — a 60%+ drop in vertical speed — so mild
        speed fluctuations mid-trajectory don't trigger this path.

        No snap is applied: this detection marks the START of the
        deceleration, not a local y-maximum, so snapping to the nearest
        y-peak would move the frame to a later, deeper position.

        Shot suppression is intentionally skipped (same reasoning as
        Path 4): low-angle bounces frequently occur just before a shot,
        and suppression would always eliminate them.  The global x-flip
        check is the sole gate.
        """
        n = len(df)
        y  = df['mid_y'].values
        dy = df['delta'].values

        valid_y = y[~np.isnan(y)]
        if len(valid_y) == 0:
            return []

        court_threshold_50 = float(np.percentile(valid_y, 50))
        min_pre_speed = self._y_range * self.decel_min_speed_frac

        back = self.decel_back
        fwd  = self.decel_fwd

        candidates = []
        for i in range(back, n - fwd):
            yi = y[i]
            if np.isnan(yi) or yi < court_threshold_50:
                continue

            pre  = [dy[j] for j in range(i - back, i) if not np.isnan(dy[j])]
            post = [dy[j] for j in range(i, i + fwd)  if not np.isnan(dy[j])]

            if len(pre) < 3 or len(post) < 3:
                continue

            pre_mean  = float(np.mean(pre))
            post_mean = float(np.mean(post))

            if pre_mean < min_pre_speed:
                continue   # ball wasn't meaningfully falling — skip

            if post_mean >= pre_mean * self.decel_ratio:
                continue   # insufficient deceleration

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
        minimum and then sharply recovers as the ball rebounds.
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
