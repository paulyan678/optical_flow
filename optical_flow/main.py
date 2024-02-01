import process_video_to_frame_image
import optical_flow
import numpy as np
import cv2
import matplotlib.pyplot as plt
if __name__ == '__main__':
    video_frames = process_video_to_frame_image.process_video_to_frame_image('homography.mp4')
    corner_locations = np.array([[247, 102], [364, 93], [378, 281], [268, 308]])
    corner_loc_all_frames = optical_flow.optical_flow_frames(video_frames, corner_locations, 15)

    # plot all corner locations onto the first frame ofthe video
    plt.imshow(video_frames[0])
    plt.scatter(corner_loc_all_frames[:, :, 0].flatten(), corner_loc_all_frames[:, :, 1].flatten(), c='r')
    plt.show()


