import cv2
import numpy as np
import matplotlib.image as mpimg

class Threshold:
    def __init__(self, kernel=3):
        self.sobel_kernel = kernel

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        return dir_binary

    def abs_sobel_thresh(self, img, orient='x', thresh=(30, 150)):
        # Convert to grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        # take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(v_channel, cv2.CV_64F, 1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(v_channel, cv2.CV_64F, 0, 1))

        # scale to 8-bit (0-255) then convert to type=uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # create a mask of 1's where the scaled gradient magnitude
        # is > thres_min and < thres_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # return this mask as the binary_output

        return binary_output

    def mag_thresh(self, img, mag_thresh=(0, 255), sobel_kernel=3):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        mag_binary = np.zeros_like(gradmag)
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return mag_binary

    def color_threshold(self, img):
        sthresh = (100, 255)
        vthresh = (50, 255)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        v_binary = np.zeros_like(v_channel)
        v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1

        output = np.zeros_like(s_channel)
        output[(s_binary == 1) & (v_binary == 1)] = 1
        return output

    # Define a function that thresholds the S-channel of HLS
    def hls_s(img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        # 2) Apply a threshold to the S channel
        # 3) Return a binary image of threshold result
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s = hls[:, :, 2]
        retval, s_binary = cv2.threshold(s.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        return s_binary

    # Define a function that thresholds the S-channel of HLS
    def hls_h(img, thresh=(0, 255)):
        # 1) Convert to HLS color space
        # 2) Apply a threshold to the S channel
        # 3) Return a binary image of threshold result
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        h = hls[:, :, 0]
        retval, h_binary = cv2.threshold(h.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        return h_binary

    def gaussian_blur(self, img, kernel=5):
        return cv2.GaussianBlur(img, (kernel, kernel), 0)

    def yello_white_threshold(self, img):
        low_y_w = np.array([140, 140, 64])
        high_y_w = np.array([255, 255, 255])
        mask = cv2.inRange(img, low_y_w, high_y_w)
        return cv2.bitwise_and(img, img, mask=mask)

    def combined_threshold(self, img):
        yw = self.yello_white_threshold(self.gaussian_blur(img, kernel=5))
        directional = self.dir_threshold(yw, sobel_kernel=15, thresh=(0.7, 1.3))
        mag = self.mag_thresh(yw, mag_thresh=(70, 255), sobel_kernel=9)
        gradx = self.abs_sobel_thresh(yw, orient='x', thresh=(25, 100))
        grady = self.abs_sobel_thresh(yw, orient='y', thresh=(50, 150))
        # sch = self.hls_s(yw, thresh=(88, 190))
        # hch = self.hls_h(yw, thresh=(50, 100))
        combined_threshold = np.zeros_like(directional).astype(np.uint8)
        combined_threshold[((gradx > 0) | (grady > 0) | (yw == 1) & ((mag == 1) | (directional == 1)))] = 1
        return combined_threshold

