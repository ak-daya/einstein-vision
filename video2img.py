from tqdm import tqdm
import cv2
from pathlib import Path

def extract_frames_from_video(path_src, dir_output):
    """
    Extract one frame for every 10 from a video and save 
    in a directory. 
    """
    if not path_src.is_file():
        print(f"File doesn't exist: {path_src}")
        return None

    # Open the video file
    cap = cv2.VideoCapture(str(path_src))
    frame_count = 0

    # Read frames until the video is finished
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as an image
        if frame_count % 10 == 0:
            path_output = dir_output / f"{int(frame_count/10)}.jpg"
            cv2.imwrite(str(path_output), frame)

        # Increment frame count
        frame_count += 1

    # Release video capture object
    cap.release()

def main():
    # cam_types = ['front', 'back', 'left', 'right']
    cam_types = ['front']
    
    # Loop over 13 scenes
    for i in tqdm(range(1, 14)):

        # Loop over all cam types (front, back, ...)
        for cam_type in cam_types:
            scene = f"scene{i}"
            src_path = Path.cwd() / "data" / scene / "Undist" / f"{cam_type}.mp4"
            base_path = Path.cwd() / "scenes" / scene

            raw_imgs_folder = base_path / f"{cam_type}" / "raw" 
            renders_folder = base_path / f"{cam_type}" / "renders"
            bb_folder = base_path / f"{cam_type}" / "boundingbox"
            depth_folder = base_path / f"{cam_type}" / "depth"

            raw_imgs_folder.mkdir(parents=True, exist_ok=True)
            renders_folder.mkdir(parents=True, exist_ok=True)
            bb_folder.mkdir(parents=True, exist_ok=True)
            depth_folder.mkdir(parents=True, exist_ok=True)

            extract_frames_from_video(src_path, raw_imgs_folder)

if __name__=="__main__":
    main()