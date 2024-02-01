import cv2
import numpy as np
def process_video_to_frame_image(video_path):
    """
    This function will process the video and convert it to frame images.
    :param video_path: The path of the video.
    :return: numpy array of gram image. The shape is (frame_num, height, width, channel).
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        return None
    
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    channel = 3
    frame_images = np.zeros((frame_num, height, width, channel), dtype=np.uint8)
    for i in range(frame_num):
        ret, frame = cap.read()
        frame_images[i] = frame
    return frame_images

if __name__ == "__main__":
    video_path = "./Question5_homography/input_video/homography.mp4"
    frame_images = process_video_to_frame_image(video_path)
    print(frame_images.shape)