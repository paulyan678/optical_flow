import numpy as np
import cv2
from scipy.signal import correlate2d
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
SOBEL_Y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

def optical_flow(frame1, frame2, keypoint, window_size):
    """
    :param frame1: first frame (height, width, channel)
    :param frame2: second frame (height, width, channel)
    :param keypoint: location of the keypoint on the first frame (2)
    :param window_size: size of the neighborhood
    :return: location of the keypoint on the second frame (2)
    """
    left = keypoint[0]-window_size
    left = left if left >= 0 else 0
    right = keypoint[0]+window_size+1
    right = right if right <= frame1.shape[0] else frame1.shape[0]
    top = keypoint[1]-window_size
    top = top if top >= 0 else 0
    bottom = keypoint[1]+window_size+1
    bottom = bottom if bottom <= frame1.shape[1] else frame1.shape[1]
    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    Ix = correlate2d(frame1_gray, SOBEL_X, mode='same')
    Iy = correlate2d(frame1_gray, SOBEL_Y, mode='same')
    It = frame2_gray - frame1_gray

    left, right, top, bottom = int(left), int(right), int(top), int(bottom)

    Ix_roi = Ix[left:right, top:bottom]
    Iy_roi = Iy[left:right, top:bottom]
    It_roi = It[left:right, top:bottom]

    Ixx_roi = Ix_roi*Ix_roi
    Iyy_roi = Iy_roi*Iy_roi
    Ixy_roi = Ix_roi*Iy_roi
    Ixt_roi = Ix_roi*It_roi
    Iyt_roi = Iy_roi*It_roi


    A = np.array([[np.sum(Ixx_roi), np.sum(Ixy_roi)],
                    [np.sum(Ixy_roi), np.sum(Iyy_roi)]])
    
    b = np.array([[-np.sum(Ixt_roi)],
                    [-np.sum(Iyt_roi)]])
    
    try:
        v = np.linalg.solve(A, b)
    except:
        print("Singular matrix")
        v = np.array([[0], [0]])

    return np.round(keypoint + v.flatten()).astype(int)


def optical_flow_frames(video_frames, keypints_location, window_size):
    """
    :param video_frames: list of frames (num grames, height, width, channel)
    :param keypints_location: list of keypoints location on the first frame (num_keypoints, 2)
    :param window_size: size of the neighborhood
    :return: list of key points location (num frames, num keypoints, 2)
    """
    key_point_locations = np.zeros((len(video_frames), len(keypints_location), 2))
    key_point_locations[0] = keypints_location


    for i, frame in enumerate(video_frames[:-1]):
        # for j, keypoint in enumerate(key_point_locations[i]):
            # key_point_locations[i+1][j] = optical_flow(video_frames[i], video_frames[i+1], keypoint, window_size)
            #use optical flow from open cv
        p1, st, err = cv2.calcOpticalFlowPyrLK(video_frames[i], video_frames[i+1], key_point_locations[i].reshape(-1, 1, 2).astype(np.float32), None, winSize=(window_size, window_size))
        key_point_locations[i+1] = p1.reshape(4, 2)

    
    return key_point_locations
