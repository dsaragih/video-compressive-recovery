import cv2
import numpy as np

def sample_frames_from_video(video_path, k, div):
    # video_path: path to the video file
    # k: number of frames to sample
    # div: number of frames to skip between each sampled frame
    # Returns: numpy array of shape (k, H, W) where H, W are frame dimensions

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    # List to store frames
    frames = []
    
    # Read the first k frames, skipping div frames between each sampled frame
    for i in range(k):
        
        # Skip div frames
        for _ in range(div - 1):
            ret, frame = cap.read()
            if not ret:
                # End of video
                break

        # Read the frame
        ret, frame = cap.read()

        if not ret:
            # End of video
            break
        
        frame = cv2.resize(frame, (320, 320))
        # Convert the frame from BGR to grayscale (if required by your program)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Append the frame to the list
        frames.append(frame_gray)

    # Release the video capture object
    cap.release()
    
    # Convert the list of frames to a numpy array for further processing
    frames_array = np.array(frames)  # Shape will be (k, H, W) where H, W are frame dimensions
    
    return frames_array

# Usage example:
video_path = "./inputs/fan.mp4"
k = 16  # Number of frames to sample
div = 6  # divide fps by div
input_imgs = sample_frames_from_video(video_path, k, div)
print(f"Sampled frames shape: {input_imgs.shape}")

# Save to /inputs directory as .png files
for i, frame in enumerate(input_imgs):
    cv2.imwrite(f"./inputs/frame_{str(i).zfill(5)}.png", frame)