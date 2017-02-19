import pickle
import cv2
import numpy as np
import matplotlib.image as mpimg
import glob

class Undistort:
    def __init__(self):
        self.mtx = []
        self.dist = []
        self.ret = []
        self.rvecs = []
        self.tvecs = []

    def undistort(self, img):
        undist_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist_img

    def calibratie_camera(self):
        nx = 9
        ny = 6
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('./camera_cal/calibration*.jpg')
        for idx, fname in enumerate(images):
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                print('Working on Image : ', fname)
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                write_name = './camera_cal/corners_found_' + str(idx) + '.jpg'
                cv2.imwrite(write_name, img)

        # Load image for reference
        img = cv2.imread('./camera_cal/calibration1.jpg')
        img_size = (img.shape[1], img.shape[0])

        # The camera calibration given object points and image points
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        return self

