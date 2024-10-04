import numpy as np
import cv2
import os
from PIL import Image


def draw_connection(image, cnt1, cnt2, layout = {"radius": 4, "color": (0, 255, 0), "line_color": (0, 255, 0), "thickness": 2}):

    # Draw the two circles
    cv2.circle(image, (cnt1[0], cnt1[1]), layout["radius"], layout["color"], cv2.FILLED)
    cv2.circle(image, (cnt2[0], cnt2[1]), layout["radius"], layout["color"], cv2.FILLED)

    # Draw a line connecting the centers of the circles
    cv2.line(image, cnt1, cnt2, layout["line_color"], layout["thickness"])

    return image



structure = [ "origin", "aspect_ratio", "Nose", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist",
    "L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle", "L_heel", "R_heel", "L_toe", "R_toe"]


def main(project, width = 400, height = 400):
    
    video_output = np.load(f"./input/features/{project}.npy")

    for i in range(video_output.shape[0]):

        info = video_output[i]
        img = np.zeros((width, height, 3)).astype(np.uint8)
        # img = Image.fromarray(canvas)

        # take out of bounding box space
        # info[2:, 0] = bbox_height * info[2:, 0]
        # info[2:, 1] = bbox_width * info[2:, 1]

        # take out offset for bounding box
        info[2:, 0] = info[2:, 0] * height/2
        info[2:, 1] = info[2:, 1] * width/2

        info = info.astype(np.uint8)

        img = draw_connection(img, info[2], info[3])
        img = draw_connection(img, info[2], info[4])
        img = draw_connection(img, info[3], info[4])

        img = draw_connection(img, info[3], info[5])
        img = draw_connection(img, info[4], info[6])
        img = draw_connection(img, info[5], info[7])
        img = draw_connection(img, info[6], info[8])

        img = draw_connection(img, info[4], info[10])
        img = draw_connection(img, info[3], info[9])
        img = draw_connection(img, info[9], info[10])

        img = draw_connection(img, info[10], info[12])
        img = draw_connection(img, info[9], info[11])
        img = draw_connection(img, info[12], info[14])
        img = draw_connection(img, info[11], info[13])
        img = draw_connection(img, info[14], info[16])
        img = draw_connection(img, info[13], info[15])

        if os.path.exists(f"./output/{project}") == False:
            os.mkdir(f"./output/{project}")

        cv2.imwrite(f"./output/{project}/frame_{i}.jpg", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

main("SixStepPeter")