# Dataset https://github.com/TuSimple/tusimple-benchmark/issues/3
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys

def preprocessing(img):
    if len(img.shape) > 2: 
        #convert into hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # white bound
        w_lower_bound = np.array([0,0,0], dtype=np.uint8)
        w_upper_bound = np.array([0,0,255], dtype=np.uint8)
        
        # yellow bound
        y_lower_bound = np.array([20,100,100], dtype=np.uint8)
        y_upper_bound = np.array([30,255,255], dtype=np.uint8)

        w_mask = cv2.inRange(hsv, w_lower_bound, w_upper_bound)
        y_mask = cv2.inRange(hsv, y_lower_bound, y_upper_bound)

        w_res = cv2.bitwise_and(img, img, mask=w_mask)
        y_res = cv2.bitwise_and(img, img, mask=y_mask)

        img = img + w_res + y_res

        return img

def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


if __name__ == "__main__":
    test_image = sys.argv[1]
    i1, i2 = preprocessing(cv2.imread(test_image))
    cv2.imshow("i1", i1)
    cv2.waitKey(0)
    cv2.destroyWindow('i1')
    cv2.imshow("i2", i2)
    cv2.waitKey(0)
    cv2.destroyWindow('i2')
