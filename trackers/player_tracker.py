# Import All the Required Libraries
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from utils import measure_distance, get_center_bbox


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # --- Step 1: Pick starting IDs from frame 0 ---
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)

        # player_half: { current_id -> 'top' | 'bottom' }
        # The half assignment travels with the player across ID changes.
        player_half = self._assign_halves(court_keypoints, player_detections_first_frame, chosen_players)

        filtered_player_detections = []

        # half_timeline: one entry per frame, maps half -> {'pid': id, 'bbox': bbox|None}
        # Tracks the current placeholder ID and detected bbox for each player slot.
        # Used by _interpolate_gaps to fill across ID changes correctly.
        half_timeline = []

        for player_det in player_detections:
            filtered_player_dict = {}
            new_chosen = []
            new_half   = {}
            frame_halves = {}

            for cid in chosen_players:
                half = player_half[cid]

                if cid in player_det:
                    # ID is present — locked on, track normally
                    bbox = player_det[cid]
                    filtered_player_dict[cid] = bbox
                    new_chosen.append(cid)
                    new_half[cid] = half
                    frame_halves[half] = {'pid': cid, 'bbox': bbox}

                else:
                    # ID is missing — search every gap frame until re-acquired.
                    # Exclude the other player's ID so we never swap halves.
                    already_tracked = set(chosen_players) - {cid}
                    new_id = self.find_player_in_half(
                        court_keypoints, player_det, half,
                        exclude_ids=already_tracked
                    )
                    if new_id is not None:
                        # Found a valid player-sized detection — lock on to new ID
                        bbox = player_det[new_id]
                        filtered_player_dict[new_id] = bbox
                        new_chosen.append(new_id)
                        new_half[new_id] = half
                        frame_halves[half] = {'pid': new_id, 'bbox': bbox}
                    else:
                        # Nobody valid found — keep placeholder ID, slot empty, try next frame
                        new_chosen.append(cid)
                        new_half[cid] = half
                        frame_halves[half] = {'pid': cid, 'bbox': None}

            chosen_players = new_chosen
            player_half    = new_half
            filtered_player_detections.append(filtered_player_dict)
            half_timeline.append(frame_halves)

        # Fill empty slots with interpolated bboxes so pose tracker has
        # a valid region for every frame, even during detection gaps.
        return self._interpolate_gaps(filtered_player_detections, half_timeline)

    def _interpolate_gaps(self, filtered_player_detections, half_timeline):
        """
        Interpolates per player half (top/bottom) rather than per track ID.
        This correctly handles gaps that span an ID change — e.g. ID 192
        disappears at frame 490 and ID 348 appears at frame 507. The gap
        frames get interpolated bboxes written under whatever placeholder ID
        is current for that half in that frame.
        """
        n = len(filtered_player_detections)

        for half in ('top', 'bottom'):
            i = 0
            while i < n:
                entry = half_timeline[i].get(half, {})

                if entry.get('bbox') is None:
                    # Find the last frame with a real bbox for this half
                    start = i - 1
                    while start >= 0 and half_timeline[start].get(half, {}).get('bbox') is None:
                        start -= 1

                    # Find the next frame with a real bbox for this half
                    end = i
                    while end < n and half_timeline[end].get(half, {}).get('bbox') is None:
                        end += 1

                    if start < 0 or end >= n:
                        # Gap at the very start or end — no anchor on one side, skip
                        i = (end + 1) if end < n else n
                        continue

                    bbox_start = np.array(half_timeline[start][half]['bbox'])
                    bbox_end   = np.array(half_timeline[end][half]['bbox'])
                    gap_len    = end - start  # total steps across the gap

                    for offset in range(1, gap_len):
                        t = offset / gap_len
                        interpolated = (bbox_start + t * (bbox_end - bbox_start)).tolist()
                        # Write under the placeholder ID current for this frame and half
                        pid = half_timeline[start + offset][half]['pid']
                        filtered_player_detections[start + offset][pid] = interpolated

                    i = end + 1
                else:
                    i += 1

        return filtered_player_detections

    def _assign_halves(self, court_keypoints, player_dict, chosen_players):
        """
        Return { track_id -> 'top' | 'bottom' } based on each player's
        position relative to the court midpoint.
        """
        ys = [court_keypoints[i] for i in range(1, len(court_keypoints), 2)]
        court_mid_y = (min(ys) + max(ys)) / 2

        player_half = {}
        for track_id in chosen_players:
            if track_id not in player_dict:
                continue
            x1, y1, x2, y2 = player_dict[track_id]
            cy = (y1 + y2) / 2
            player_half[track_id] = 'top' if cy < court_mid_y else 'bottom'
        return player_half

    def find_player_in_half(self, court_keypoints, player_dict, half,
                             exclude_ids=None, min_bbox_area=8000):
        """
        Find the largest valid detection strictly within the requested half.
        Uses a high min_bbox_area (10000) to exclude ballboys and other
        small non-player detections.
        Does NOT fall back to the other half — returns None if nothing is found.
        Excludes any IDs in exclude_ids (i.e. the other player already tracked).
        """
        if exclude_ids is None:
            exclude_ids = set()

        xs = [court_keypoints[i] for i in range(0, len(court_keypoints), 2)]
        ys = [court_keypoints[i] for i in range(1, len(court_keypoints), 2)]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        court_mid_y  = (min_y + max_y) / 2
        court_width  = max_x - min_x
        court_height = max_y - min_y

        corner_x_thresh = 0.05 * court_width
        mid_y_thresh    = 0.08 * court_height

        candidates = []

        for track_id, bbox in player_dict.items():
            if track_id in exclude_ids:
                continue
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area < min_bbox_area:
                continue
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if cx < min_x + corner_x_thresh or cx > max_x - corner_x_thresh:
                continue
            if abs(cy - court_mid_y) < mid_y_thresh:
                continue
            det_half = 'top' if cy < court_mid_y else 'bottom'
            if det_half != half:
                continue
            candidates.append((track_id, area))

        if not candidates:
            return None

        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def choose_players(self, court_keypoints, player_dict, min_bbox_area: int = 5000):
        """
        Pick one player from the top half and one from the bottom half of the court.
        Used for the initial frame-0 ID selection.
        """
        xs = [court_keypoints[i] for i in range(0, len(court_keypoints), 2)]
        ys = [court_keypoints[i] for i in range(1, len(court_keypoints), 2)]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        court_mid_y  = (min_y + max_y) / 2
        court_width  = max_x - min_x
        court_height = max_y - min_y

        corner_x_thresh = 0.05 * court_width
        mid_y_thresh    = 0.08 * court_height

        print(f"[choose_players] court bounds: x=[{min_x:.0f},{max_x:.0f}] y=[{min_y:.0f},{max_y:.0f}] mid_y={court_mid_y:.0f}")
        print(f"[choose_players] thresholds: corner_x={corner_x_thresh:.0f} mid_y={mid_y_thresh:.0f} min_area={min_bbox_area}")
        print(f"[choose_players] raw detections in frame 0: {len(player_dict)}")

        top_candidates    = []
        bottom_candidates = []

        for track_id, bbox in player_dict.items():
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            fail_area   = area < min_bbox_area
            fail_corner = cx < min_x + corner_x_thresh or cx > max_x - corner_x_thresh
            fail_mid    = abs(cy - court_mid_y) < mid_y_thresh
            print(f"  id={track_id} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) area={area:.0f} cx={cx:.0f} cy={cy:.0f} | fail_area={fail_area} fail_corner={fail_corner} fail_mid={fail_mid}")
            if fail_area or fail_corner or fail_mid:
                continue
            if cy < court_mid_y:
                top_candidates.append((track_id, area))
            else:
                bottom_candidates.append((track_id, area))

        top_candidates.sort(key=lambda x: -x[1])
        bottom_candidates.sort(key=lambda x: -x[1])

        print(f"[choose_players] top_candidates={top_candidates}")
        print(f"[choose_players] bottom_candidates={bottom_candidates}")

        chosen_players = []
        if top_candidates:
            chosen_players.append(top_candidates[0][0])
        if bottom_candidates:
            chosen_players.append(bottom_candidates[0][0])

        return chosen_players

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        class_names = results.names
        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            class_ids = box.cls.tolist()[0]
            det_class_names = class_names[class_ids]
            if det_class_names == "person":
                player_dict[track_id] = result
        return player_dict

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
                return player_detections
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        return player_detections

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames