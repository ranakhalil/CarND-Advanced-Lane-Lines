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

    def mag_thresh(self, img, mag_thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
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

    def combined_threshold(self, img):
        directional = self.dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
        mag = self.mag_thresh(img, mag_thresh=(50, 255))
        color = self.color_threshold(img)
        # gradx = self.abs_sobel_thresh(img, orient='x', thresh=(25, 100))
        # grady = self.abs_sobel_thresh(img, orient='y', thresh=(50, 150))
        combined_threshold = np.zeros_like(directional).astype(np.uint8)
        combined_threshold[((color == 1) & ((mag == 1) | (directional == 1)))] = 1
        return combined_threshold
