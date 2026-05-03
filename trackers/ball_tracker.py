import torch
import cv2
import numpy as np
from tqdm import tqdm
from itertools import groupby
from scipy.spatial import distance
from model import BallTrackerNet
from general import postprocess


class BallTracker:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()

        self.infer_width = 640
        self.infer_height = 360
        self.train_width = 1280
        self.train_height = 720

    def detect_frames(self, frames, extrapolation=True):
        ball_track, dists = self._infer(frames)
        ball_track = self._remove_outliers(ball_track, dists)
        if extrapolation:
            none_frames = [i for i, (x, y) in enumerate(ball_track) if x is None]
            print("None frames before split:", none_frames)
            subtracks = self._split_track(ball_track)
            for r in subtracks:
                ball_subtrack = ball_track[r[0]:r[1]]
                ball_subtrack = self._interpolation(ball_subtrack)
                ball_track[r[0]:r[1]] = ball_subtrack

        return ball_track

    def draw_bboxes(self, frames, ball_track, trace=7):
        output_frames = []
        for num, frame in enumerate(frames):
            frame = frame.copy()
            for i in range(trace):
                if (num - i) >= 0 and (num - i) < len(ball_track):
                    if ball_track[num - i][0] is not None:
                        x = int(ball_track[num - i][0])
                        y = int(ball_track[num - i][1])
                        thickness = max(1, 10 - i)
                        frame = cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=thickness)
                    else:
                        break
            output_frames.append(frame)
        return output_frames

    def _infer(self, frames):
        dists = [-1] * 2
        ball_track = [(None, None)] * 2

        orig_height, orig_width = frames[0].shape[:2]
        scale_x = orig_width / self.train_width
        scale_y = orig_height / self.train_height

        for num in tqdm(range(2, len(frames)), desc="Tracking ball"):
            img = cv2.resize(frames[num], (self.infer_width, self.infer_height))
            img_prev = cv2.resize(frames[num - 1], (self.infer_width, self.infer_height))
            img_preprev = cv2.resize(frames[num - 2], (self.infer_width, self.infer_height))

            imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
            imgs = imgs.astype(np.float32) / 255.0
            imgs = np.rollaxis(imgs, 2, 0)
            inp = np.expand_dims(imgs, axis=0)

            out = self.model(torch.from_numpy(inp).float().to(self.device))
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = postprocess(output)

            if x_pred is not None and y_pred is not None:
                x_pred = int(x_pred * scale_x)
                y_pred = int(y_pred * scale_y)

            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] and ball_track[-2][0]:
                dist = distance.euclidean(ball_track[-1], ball_track[-2])
            else:
                dist = -1
            dists.append(dist)
        return ball_track, dists

    def _remove_outliers(self, ball_track, dists, max_dist=100, context_window=10, max_context_dist=50):
        """
        Extends the original outlier removal with a context window check.

        For isolated detections — where both immediate neighbors are None and therefore
        dists are -1 on both sides — the original logic never flags them as outliers at
        all because neither dist exceeds max_dist. This pass handles that blind spot.

        For each isolated detection we search outward up to `context_window` frames in
        both directions to find the nearest real detections before and after the gap.
        We then ask: is this isolated point plausible given the trajectory implied by
        those anchor detections?

        Plausibility is measured by projecting linearly from the two anchors to estimate
        where the ball *should* be at the isolated frame, then checking whether the actual
        detection is within `max_context_dist` pixels of that estimate. If no anchor
        exists on one side, we fall back to a simple distance check from whichever anchor
        we do have.

        If the isolated detection fails this check, it is wiped to (None, None).
        """
        # Run the original distance-based outlier removal first
        outliers = list(np.where(np.array(dists) > max_dist)[0])
        for i in outliers:
            if (dists[i + 1] > max_dist) | (dists[i + 1] == -1):
                ball_track[i] = (None, None)
                outliers.remove(i)
            elif dists[i - 1] == -1:
                ball_track[i - 1] = (None, None)

        # Context window pass: catch isolated detections surrounded by blanks.
        # Runs repeatedly until no new frames are wiped so that a bad isolated
        # detection cannot shield a neighbour by acting as a plausible anchor.
        while True:
            wiped_any = False
            for i in range(len(ball_track)):
                if ball_track[i][0] is None:
                    continue

                prev_none = (i == 0) or (ball_track[i - 1][0] is None)
                next_none = (i == len(ball_track) - 1) or (ball_track[i + 1][0] is None)

                if not (prev_none and next_none):
                    # Not isolated — already handled by the distance-based pass above
                    continue

                # Find the nearest real detection before i
                anchor_before = None
                anchor_before_idx = None
                for j in range(i - 1, max(i - 1 - context_window, -1), -1):
                    if ball_track[j][0] is not None:
                        anchor_before = ball_track[j]
                        anchor_before_idx = j
                        break

                # Find the nearest real detection after i
                anchor_after = None
                anchor_after_idx = None
                for j in range(i + 1, min(i + 1 + context_window, len(ball_track))):
                    if ball_track[j][0] is not None:
                        anchor_after = ball_track[j]
                        anchor_after_idx = j
                        break

                if anchor_before is None and anchor_after is None:
                    # No context at all — cannot evaluate, leave it
                    continue

                if anchor_before is not None and anchor_after is not None:
                    # Linearly interpolate between the two anchors to get an expected position
                    t_before = anchor_before_idx
                    t_after = anchor_after_idx
                    t = i
                    alpha = (t - t_before) / (t_after - t_before)
                    expected_x = anchor_before[0] + alpha * (anchor_after[0] - anchor_before[0])
                    expected_y = anchor_before[1] + alpha * (anchor_after[1] - anchor_before[1])
                    expected = (expected_x, expected_y)
                elif anchor_before is not None:
                    # Only a before-anchor: check raw distance from it, scaled by frame gap
                    gap = i - anchor_before_idx
                    expected = anchor_before
                    max_context_dist_scaled = max_context_dist * gap
                    if distance.euclidean(ball_track[i], expected) > max_context_dist_scaled:
                        ball_track[i] = (None, None)
                        wiped_any = True
                    continue
                else:
                    # Only an after-anchor: same idea
                    gap = anchor_after_idx - i
                    expected = anchor_after
                    max_context_dist_scaled = max_context_dist * gap
                    if distance.euclidean(ball_track[i], expected) > max_context_dist_scaled:
                        ball_track[i] = (None, None)
                        wiped_any = True
                    continue

                if distance.euclidean(ball_track[i], expected) > max_context_dist:
                    ball_track[i] = (None, None)
                    wiped_any = True
            if not wiped_any:
                break

        # Paired outlier pass: catch two consecutive detections surrounded by blanks.
        # Each point is evaluated independently by pretending its partner doesn't exist.
        for i in range(len(ball_track) - 1):
            if ball_track[i][0] is None or ball_track[i + 1][0] is None:
                continue

            prev_none = (i == 0) or (ball_track[i - 1][0] is None)
            next_none = (i + 1 == len(ball_track) - 1) or (ball_track[i + 2][0] is None)

            if not (prev_none and next_none):
                continue

            # Evaluate each point of the pair as if it were isolated
            for idx in [i, i + 1]:
                if ball_track[idx][0] is None:
                    continue  # already wiped by earlier evaluation of the other point

                # Find nearest real detection before idx, skipping the partner
                anchor_before = None
                anchor_before_idx = None
                for j in range(idx - 1, max(idx - 1 - context_window, -1), -1):
                    if j == (i + 1 if idx == i else i):
                        continue  # skip the partner
                    if ball_track[j][0] is not None:
                        anchor_before = ball_track[j]
                        anchor_before_idx = j
                        break

                # Find nearest real detection after idx, skipping the partner
                anchor_after = None
                anchor_after_idx = None
                for j in range(idx + 1, min(idx + 1 + context_window, len(ball_track))):
                    if j == (i + 1 if idx == i else i):
                        continue  # skip the partner
                    if ball_track[j][0] is not None:
                        anchor_after = ball_track[j]
                        anchor_after_idx = j
                        break

                if anchor_before is None and anchor_after is None:
                    continue

                if anchor_before is not None and anchor_after is not None:
                    alpha = (idx - anchor_before_idx) / (anchor_after_idx - anchor_before_idx)
                    expected_x = anchor_before[0] + alpha * (anchor_after[0] - anchor_before[0])
                    expected_y = anchor_before[1] + alpha * (anchor_after[1] - anchor_before[1])
                    expected = (expected_x, expected_y)
                    if distance.euclidean(ball_track[idx], expected) > max_context_dist:
                        ball_track[idx] = (None, None)
                elif anchor_before is not None:
                    gap = idx - anchor_before_idx
                    if distance.euclidean(ball_track[idx], anchor_before) > max_context_dist * gap:
                        ball_track[idx] = (None, None)
                else:
                    gap = anchor_after_idx - idx
                    if distance.euclidean(ball_track[idx], anchor_after) > max_context_dist * gap:
                        ball_track[idx] = (None, None)
        # Spike pass: single detection flanked by two real neighbors where both distances
        # exceed max_dist.  The first-pass loop misses this when the list-mutation-during-
        # iteration bug causes the frame to be skipped, and the isolated/paired passes skip
        # it because neither neighbor is None.
        for i in range(1, len(ball_track) - 1):
            if ball_track[i][0] is None:
                continue
            if ball_track[i - 1][0] is None or ball_track[i + 1][0] is None:
                continue
            d_prev = distance.euclidean(ball_track[i - 1], ball_track[i])
            d_next = distance.euclidean(ball_track[i], ball_track[i + 1])
            if d_prev > max_dist and d_next > max_dist:
                ball_track[i] = (None, None)

        # Startup prefix pass: catch 1–2 bogus frames at the start of a consecutive run
        # (right after a None gap) that are internally coherent but make a large jump into
        # the real tracking that follows them in the same run.  The previous small-group
        # pass failed here because frames 108-109 are immediately followed by real tracking
        # at 110+, so the group-finding loop walked past them and the group size exceeded 2.
        for i in range(len(ball_track)):
            if ball_track[i][0] is None:
                continue
            if i > 0 and ball_track[i - 1][0] is not None:
                continue  # not the start of a post-gap group

            g_end = i
            while g_end + 1 < len(ball_track) and ball_track[g_end + 1][0] is not None:
                g_end += 1
            for p in range(1, 3):  # try prefix lengths 1 and 2
                split = i + p
                if split >= len(ball_track) or ball_track[split][0] is None:
                    break  # not enough real frames to compare against
                # prefix must be internally coherent (no large jumps within it)
                prefix_coherent = all(
                    distance.euclidean(ball_track[k], ball_track[k + 1]) <= max_dist
                    for k in range(i, split - 1)
                )
                if not prefix_coherent:
                    break
                jump = distance.euclidean(ball_track[split - 1], ball_track[split])
                if jump > max_dist * 3:
                    for k in range(i, split):
                        ball_track[k] = (None, None)
                    break

        # Tail suffix pass: catch the last detection before a gap whose extrapolated
        # velocity overshoots the next post-gap detection.  When extrap_error > 1.5x
        # actual_dist the ball would have needed to reverse course to reach its next
        # observed position — strong evidence the tail detection is a false positive
        # riding ahead of the true trajectory rather than a legitimate pre-bounce frame.
        # A ball that was simply hit hard away from its current direction produces an
        # extrap_error that is at most comparable to actual_dist (not much larger).
        for i in range(len(ball_track) - 1):
            if ball_track[i][0] is None:
                continue
            if ball_track[i + 1][0] is not None:
                continue  # not the last frame before a gap
            if i == 0 or ball_track[i - 1][0] is None:
                continue  # need a prior detection to estimate velocity

            # Find first detection after the gap (within context_window)
            j = i + 1
            while j < len(ball_track) and ball_track[j][0] is None:
                j += 1
            if j >= len(ball_track) or (j - i) > context_window:
                continue

            gap = j - i
            vx = ball_track[i][0] - ball_track[i - 1][0]
            vy = ball_track[i][1] - ball_track[i - 1][1]

            ex = ball_track[i][0] + gap * vx
            ey = ball_track[i][1] + gap * vy

            extrap_error = distance.euclidean((ex, ey), ball_track[j])
            actual_dist = distance.euclidean(ball_track[i], ball_track[j])

            if extrap_error > actual_dist * 1.5 and extrap_error > max_dist:
                ball_track[i] = (None, None)

        return ball_track

    def _split_track(self, ball_track, max_gap=8, max_dist_gap=80, min_track=5):
        list_det = [0 if x[0] else 1 for x in ball_track]
        groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]
        cursor = 0
        min_value = 0
        result = []
        for i, (k, l) in enumerate(groups):
            if (k == 1) & (i > 0) & (i < len(groups) - 1):
                dist = distance.euclidean(ball_track[cursor - 1], ball_track[cursor + l])
                if (l >= max_gap) | (dist / l > max_dist_gap):
                    if cursor - min_value > min_track:
                        result.append([min_value, cursor])
                        min_value = cursor + l - 1
            cursor += l
        if len(list_det) - min_value > min_track:
            result.append([min_value, len(list_det)])
        return result

    def _interpolation(self, coords):
        def nan_helper(y):
            return np.isnan(y), lambda z: z.nonzero()[0]

        x = np.array([c[0] if c[0] is not None else np.nan for c in coords])
        y = np.array([c[1] if c[1] is not None else np.nan for c in coords])

        nons, yy = nan_helper(x)
        x[nons] = np.interp(yy(nons), yy(~nons), x[~nons])
        nans, xx = nan_helper(y)
        y[nans] = np.interp(xx(nans), xx(~nans), y[~nans])

        return [*zip(x, y)]