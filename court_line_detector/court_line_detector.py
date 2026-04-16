#Import All the Required Libraries
import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
import numpy as np

# Each tuple is (keypoint_index_A, keypoint_index_B, label)
COURT_LINES = [
    (0, 1,  "back baseline"),
    (1, 3,  "right sideline"),
    (2, 3,  "front baseline"),
    (0, 2,  "left sideline"),
    (4, 5,  "left alley"),
    (6, 7,  "right alley"),
    (10, 11, "front service line"),
    (8, 9,  "back service line"),
    (12, 13, "service box divider"),
]
 
 
def _kp(keypoints, idx):
    """Return (x, y) integer tuple for keypoint index `idx`."""
    return (int(keypoints[idx * 2]), int(keypoints[idx * 2 + 1]))

# Real-world court coordinates (in feet, top-down view)
# These must match the order of YOUR model's 14 keypoints exactly —
# adjust once you've confirmed the keypoint layout.
COURT_REAL_WORLD_KEYPOINTS = np.array([
    [0,   0],   # 0  - back-left baseline corner
    [36,  0],   # 1  - back-right baseline corner
    [0,  78],   # 2  - front-left baseline corner
    [36, 78],   # 3  - front-right baseline corner
    [0,   0],   # 4  - left alley back
    [0,  78],   # 5  - left alley front
    [36,  0],   # 6  - right alley back
    [36, 78],   # 7  - right alley front
    [4.5, 0],   # 8  - back service line left
    [31.5,0],   # 9  - back service line right
    [4.5,78],   # 10 - front service line left
    [31.5,78],  # 11 - front service line right
    [18,  0],   # 12 - service box centre back
    [18, 78],   # 13 - service box centre front
], dtype=np.float32)

# Net runs the full doubles width at the court midpoint (y=39)
NET_REAL_WORLD = np.array([
    [[0,  39]],   # left net post
    [[36, 39]],   # right net post
], dtype=np.float32)

def compute_net_image_points(keypoints):
    """
    Given detected image keypoints (flat array, length 28),
    return the two net post image coordinates via homography.
    """
    image_pts = keypoints.reshape(14, 2).astype(np.float32)

    H, _ = cv2.findHomography(COURT_REAL_WORLD_KEYPOINTS, image_pts)

    net_image_pts = cv2.perspectiveTransform(NET_REAL_WORLD, H)
    left_post  = tuple(net_image_pts[0][0].astype(int))
    right_post = tuple(net_image_pts[1][0].astype(int))
    return left_post, right_post

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained = True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location = "cpu"))
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_height, original_width = image.shape[:2]
        keypoints[::2] *= original_width / 224.0
        keypoints[1::2] *= original_height / 224.0
        return keypoints

    #Plot Keypoints on the image
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(image, str(i//2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            cv2.circle(image, (x,y), 5, (0,0,255), -1)
        return image

    #Plot/ Draw keypoints on the video
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames


    def draw_lines(self, image, keypoints, color=(0, 255, 0), thickness=2):
        """
        Draw court lines connecting the defined keypoint pairs, plus a
        centre line connecting the midpoint of the left sideline (0-2)
        to the midpoint of the right sideline (1-3).
        """
        # Standard court lines
        for (a, b, _label) in COURT_LINES:
            cv2.line(image, _kp(keypoints, a), _kp(keypoints, b), color, thickness)

        # Net Posts
        left_post, right_post = compute_net_image_points(keypoints)
        cv2.line(image, left_post, right_post, (0, 200, 255), thickness)
 
        return image
 
    def draw_lines_on_video(self, video_frames, keypoints, color=(0, 255, 0), thickness=2):
        """Apply draw_lines to every frame and return the annotated list."""
        return [
            self.draw_lines(frame, keypoints, color=color, thickness=thickness)
            for frame in video_frames
        ]