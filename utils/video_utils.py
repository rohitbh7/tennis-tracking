import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

#def save_video(output_video_frames, output_video_path):
#    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#    out=cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
#    for frame in output_video_frames:
#        out.write(frame)
#    out.release()

def draw_axes(frames):
    output = []

    for frame in frames:
        h, w = frame.shape[:2]
        img = frame.copy()

        # X axis (horizontal)
        cv2.line(img, (0, 0), (w, 0), (255, 0, 0), 2)
        cv2.putText(img, "X →", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Y axis (vertical)
        cv2.line(img, (0, 0), (0, h), (0, 255, 0), 2)
        cv2.putText(img, "Y ↓", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # origin
        cv2.circle(img, (0, 0), 6, (0, 0, 255), -1)
        cv2.putText(img, "(0,0)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        output.append(img)

    return output

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed from MJPG
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()