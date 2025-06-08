import cv2


def read_video(video_path):
    """
    Reads a video file and returns a VideoCapture object.
    
    Args:
        video_path (str): Path to the video file.
        
    Returns:
        cv2.VideoCapture: VideoCapture object for the video file.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        r, frame = cap.read()
        if not r:
            break
        frames.append(frame)
    return frames


def save_video(frames, output_path, fps=30):
    """
    Saves a list of frames as a video file.
    
    Args:
        frames (list): List of frames (numpy arrays).
        output_path (str): Path to save the video file.
        fps (int): Frames per second for the video.
    """
    if not frames:
        raise ValueError("No frames to save.")
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video encoding
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()