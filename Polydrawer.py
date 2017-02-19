import numpy as np
import cv2

class Polydrawer:
    def draw(self, img, left_fit, right_fit, Minv):
        warp = np.zeros_like(img)
        fity = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
        right_fitx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_fitx, fity]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, fity])))])
        points = np.hstack((pts_left, pts_right))
        points = np.array(points, dtype=np.int32)

        cv2.fillPoly(warp, points, (0, 255, 0))

        newwarp = cv2.warpPerspective(warp, Minv, (img.shape[1], img.shape[0]))

        result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        return result