import torch
import numpy as np
import supervision as sv
from PIL import Image
from accelerate import Accelerator
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)
import cv2


class PoseDetector:
    def __init__(
        self,
        person_model_name: str = "PekingU/rtdetr_r50vd_coco_o365",
        pose_model_name: str = "usyd-community/vitpose-base-simple",
        detection_threshold: float = 0.3,
    ):
        self.device = Accelerator().device
        self.detection_threshold = detection_threshold

        # Person detector
        self.person_processor = AutoProcessor.from_pretrained(person_model_name)
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            person_model_name, device_map=self.device
        )

        # Pose estimator
        self.pose_processor = AutoProcessor.from_pretrained(pose_model_name)
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            pose_model_name, device_map=self.device
        )

    def detect_frame(self, frame: np.ndarray, player_boxes: list[list] | None = None) -> sv.KeyPoints | None:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if player_boxes is not None and len(player_boxes) > 0:
            # Skip RTDetr entirely — use tracker-provided boxes
            person_boxes = np.array(player_boxes, dtype=np.float32)
        else:
            # Fallback to full person detection if no boxes given
            inputs = self.person_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.person_model(**inputs)
            results = self.person_processor.post_process_object_detection(
                outputs,
                target_sizes=torch.tensor([(image.height, image.width)]),
                threshold=self.detection_threshold,
            )
            result = results[0]
            person_boxes = result["boxes"][result["labels"] == 0].cpu().numpy()
            if len(person_boxes) == 0:
                return None

        # VOC (x1,y1,x2,y2) → COCO (x1,y1,w,h)
        person_boxes = person_boxes.copy()
        person_boxes[:, 2] -= person_boxes[:, 0]
        person_boxes[:, 3] -= person_boxes[:, 1]

        inputs = self.pose_processor(
            image, boxes=[person_boxes], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.pose_model(**inputs)

        pose_results = self.pose_processor.post_process_pose_estimation(
            outputs, boxes=[person_boxes]
        )[0]

        xy = torch.stack([r["keypoints"] for r in pose_results]).cpu().numpy()
        scores = torch.stack([r["scores"] for r in pose_results]).cpu().numpy()
        return sv.KeyPoints(xy=xy, confidence=scores)

    def detect_frames(
        self,
        frames: list[np.ndarray],
        player_detections: list[dict] | None = None,   # pass tracker output here
    ) -> list[sv.KeyPoints | None]:
        results = []
        for i, frame in enumerate(frames):
            boxes = None
            if player_detections is not None:
                # Extract [x1,y1,x2,y2] boxes for this frame's tracked players only
                boxes = list(player_detections[i].values())
            results.append(self.detect_frame(frame, player_boxes=boxes))
        return results

    def draw_poses(
        self,
        frames: list[np.ndarray],
        pose_detections: list[sv.KeyPoints | None],
        edge_color: sv.Color = sv.Color.GREEN,
        vertex_color: sv.Color = sv.Color.RED,
        thickness: int = 4,
        radius: int = 2,
    ) -> list[np.ndarray]:
        """Annotate a list of frames with skeleton edges and joint vertices."""
        edge_annotator = sv.EdgeAnnotator(color=edge_color, thickness=thickness)
        vertex_annotator = sv.VertexAnnotator(color=vertex_color, radius=radius)

        output_frames = []
        for num, (frame, key_points) in enumerate(zip(frames, pose_detections)):
            annotated = frame.copy()
            if key_points is not None:
                annotated = edge_annotator.annotate(scene=annotated, key_points=key_points)
                annotated = vertex_annotator.annotate(scene=annotated, key_points=key_points)
            print(f"Annotated poses in frame {num}.")
            output_frames.append(annotated)

        return output_frames