import numpy as np
import cv2

def preprocessing(img):
    if len(image.shape) > 2: 
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

        return w_res, y_res 

if __name__ == "__main__":
        