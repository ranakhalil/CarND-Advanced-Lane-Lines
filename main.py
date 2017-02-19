import numpy as np
import cv2
from Undistort import Undistort
from Threshold import Threshold
from Warp import Warp
from Polyfit import Polyfit
from Polydrawer import Polydrawer
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.video.io.VideoFileClip import VideoFileClip

undistort = Undistort()
polydraw = Polydrawer()
polyfit = Polyfit()

src = np.float32([
    [580, 460],
    [700, 460],
    [1040, 680],
    [260, 680],
])

dst = np.float32([
    [260, 0],
    [1040, 0],
    [1040, 720],
    [260, 720],
])

warp = Warp(src, dst)
undistort.calibratie_camera()

def main():
    # test_images = ['straight_lines1.jpg', 'straight_lines2.jpg', 'test1.jpg',
    #                'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']
    # for image in test_images:
    #     img = mpimg.imread('./test_images/' + image)
    #     process_image(img, image)
    white_output = 'project_video_done_2.mp4'
    clip1 = VideoFileClip('project_video.mp4')
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    del clip1.reader
    del clip1


def process_image(base_image):

    img_undist = undistort.undistort(base_image)
    #mpimg.imsave('./output_images/{}_undistorted.jpg'.format(name), img_undist)

    threshold = Threshold(kernel=3)
    img = threshold.combined_threshold(img_undist)
    #mpimg.imsave('./output_images/{}_thresholded.jpg'.format(name), img, cmap="gray")

    img = warp.warp(img)
    #mpimg.imsave('./output_images/{}_warped.jpg'.format(name), img, cmap="gray")

    left_fit, right_fit = polyfit.polyfit(img, visualize=False)

    img = polydraw.draw(img_undist, left_fit, right_fit, warp.Minv)

    #mpimg.imsave('./output_images/{}_final.jpg'.format(name), img, cmap="gray")

if __name__ == '__main__':
    main()


