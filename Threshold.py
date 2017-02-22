import cv2
import numpy as np
import matplotlib.image as mpimg

class Threshold:
    def __init__(self):
        self.sobel_kernel = 15
        self.abs_sobel_thresh_x_min = 20
        self.abs_sobel_thresh_x_max = 100
        self.abs_sobel_thresh_y_min = 50
        self.abs_sobel_thresh_y_max = 150

    def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx=cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely=cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
        # create a mask of 1's where the scaled gradient magnitude
        sxbinary = np.zeros_like(dir_sobel)
        sxbinary[(dir_sobel>=thresh[0])&(dir_sobel<=thresh[1])]=1
        # return this mask as the binary_output
        binary_output = sxbinary
        return binary_output

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=5, thresh=(30, 150)):
        # Convert to grayscale
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:,:,2]
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(v_channel, cv2.CV_64F,1, 0))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(v_channel, cv2.CV_64F,0, 1))

        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # is > thres_min and < thres_max
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0])&(scaled_sobel <= thresh[1])]=1
        return binary_output

    def mag_threshold(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gradmag)
        binary_output [(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return binary_output

    def color_threshold(self, img):
        # HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # For yellow
        yellow = cv2.inRange(img, (0, 80, 200), (40, 255, 255))

        # For white
        sensitivity_1 = 10
        white = cv2.inRange(img, (0, 0, 255 - sensitivity_1), (255, 20, 255))

        sensitivity_2 = 55
        # HSL = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        white_2 = cv2.inRange(img, (0, 255-sensitivity_2, 0), (255, 80, 255))
        white_3 = cv2.inRange(img, (200, 200, 200), (255, 255, 255))

        binary_output = np.zeros_like(img[:, :, 0])
        binary_output[((yellow != 0) | (white != 0) | (white_2 != 0) | (white_3 != 0))] = 1
        return binary_output

    def lab_threshold(self, image, thresh=(0, 255)):
        lab = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2LAB)
        ch = lab[:, :, 2]
        lab_binary = np.zeros_like(ch)
        lab_binary[(ch >= thresh[0]) & (ch <= thresh[1])] = 1
        return lab_binary

    def ycrcb_threshold(self, image, thresh=(0, 255)):
        ycbcrb = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2YCR_CB)
        ch = ycbcrb[:, :, 0]
        ycrcb_binary = np.zeros_like(ch)
        ycrcb_binary[(ch >= thresh[0]) & (ch <= thresh[1])] = 1
        return ycrcb_binary

    def luv_threshold(self, image, thresh=(0, 255)):
        luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        ch = luv[:, :, 1]
        luv_binary = np.zeros_like(ch)
        luv_binary[(ch >= thresh[0]) & (ch <= thresh[1])] = 1
        return luv_binary


    def hls_s(self, img, thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s = hls[:,:,2]
        retval, s_binary = cv2.threshold(s.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        return s_binary

    def hls_h(self, img, thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        h = hls[:,:,0]
        retval, h_binary = cv2.threshold(h.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        return h_binary

    def gaussian_blur(self, img, kernel=5):
        return cv2.GaussianBlur(img, (kernel, kernel), 0)
    def apply_color_mask(hsv,img,low,high):
        # Takes in color mask and returns image with mask applied.
        mask = cv2.inRange(hsv, low, high)
        res = cv2.bitwise_and(img,img, mask= mask)
        return res

    def yello_white_threshold(self, image):

        HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # For yellow
        yellow_mask = cv2.inRange(HSV, (20, 100, 100), (50, 255, 255))
        yellow = cv2.bitwise_and(image,image, mask=yellow_mask)

        # For white
        sensitivity_1 = 75
        white_mask = cv2.inRange(HSV, (20,0,255-sensitivity_1), (255,80,255))
        white = cv2.bitwise_and(image, image, mask=white_mask)
        # sensitivity_2 = 60
        # HSL = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # white_2 = cv2.inRange(HSL, (0,255-sensitivity_2,0), (255,255,sensitivity_2))
        # white_3 = cv2.inRange(image, (200,200,200), (255,255,255))
        #
        # yw_binary = np.zeros_like(image).astype(np.uint8)
        # yw_binary[(yellow == 1) | (white == 1) | (white_2 == 1) | (white_3 == 1)] = 1
        return cv2.bitwise_or(yellow,white)


    def combined_threshold(self, image):
        # Thanks to Denise's help
        # image = self.gaussian_blur(image, kernel=5)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        yw = self.yello_white_threshold(image)
        gradx = self.abs_sobel_thresh(yw, orient='x', thresh=(20, 100))
        grady = self.abs_sobel_thresh(yw, orient='y', thresh=(20, 100))
        # Thanks for Udacity reviwer to ask me to experiment with lab
        lab_binary = self.lab_threshold(image, thresh=(150, 255))
        # thanks to Jim Winquist for inspiring me to play with YCR_CB color space
        ycrcb_binary = self.ycrcb_threshold(image, thresh=(195, 255))
        luv_binary = self.luv_threshold(image, thresh=(170, 255))
        combined = np.zeros_like(gray).astype(np.uint8)
        combined[((gradx == 1) & (grady == 1)) | (lab_binary == 1) | (ycrcb_binary == 1) | (luv_binary == 1)] = 1
        combined = (combined.copy() * 255).astype('uint8')
        # magnitude = self.mag_threshold(image, sobel_kernel=15, mag_thresh=(50, 255))
        # directional = self.dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
        # sch = self.hls_s(image, thresh=(100, 255))
        # combined = np.zeros_like(directional).astype(np.uint8)
        # combined[((color == 1) & ((magnitude == 1) | (directional == 1)))] = 1
        return combined

