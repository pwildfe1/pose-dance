from cvzone.PoseModule import PoseDetector
import cv2
import os
import numpy as np


def generate_relative(positions, img):

    pts = positions[:, 0:2]
    origin = np.array([np.min(pts[:, 0]), np.min(pts[:, 1])])

    normalized_origin = np.zeros(origin.shape)
    normalized_origin[0] = origin[0]/img.shape[0]
    normalized_origin[1] = origin[1]/img.shape[1]
    
    output = np.zeros((pts.shape[0] + 1, pts.shape[1]))
    output[0] = normalized_origin
    pts[:, 0] = pts[:, 0]/img.shape[0]
    pts[:, 1] = pts[:, 1]/img.shape[1]
    pts = pts - normalized_origin

    output[1:] = pts
    
    return output


def main(video_file):

    # Read the video from specified path 
    cam = cv2.VideoCapture(video_file)

    directory = f'./frames/{os.path.splitext(video_file)[0].split("/")[-1]}'

    if os.path.exists(directory) == False:
        os.mkdir(directory)

    # Initialize the PoseDetector class with the given parameters
    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=True,
                            enableSegmentation=False,
                            smoothSegmentation=True,
                            detectionCon=0.5,
                            trackCon=0.5)

    currentframe = 0
    # Loop to continuously get frames from the webcam
    # 0->10 (face)

    body_dict = {
        "Nose": 0,
        "L_shoulder": 11,
        "R_shoulder": 12,
        "L_elbow": 13,
        "R_elbow": 14,
        "L_wrist": 15,
        "R_wrist": 16,
        "L_hip": 23,
        "R_hip": 24,
        "L_knee": 25,
        "R_knee": 26,
        "L_ankle": 27,
        "R_ankle": 28,
        "L_heel": 29,
        "R_heel": 30,
        "L_toe": 31,
        "R_toe": 32
    }

    body_indices = []
    for k in body_dict.keys():
        body_indices.append(body_dict[k])

    frames = []

    while True:
        # Capture each frame from the webcam
        success, img = cam.read()

        if success == False: break

        # Find the human pose in the frame
        img = detector.findPose(img)

        # Find the landmarks, bounding box, and center of the body in the frame
        # Set draw=True to draw the landmarks and bounding box on the image
        lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

        # Check if any body landmarks are detected
        if lmList:

            # Get the center of the bounding box around the body
            center = bboxInfo["center"]
            cv2.circle(img, lmList[0][0:2], 8, (0,255,0), cv2.FILLED)

            # Draw a circle at the center of the bounding box
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

            frames.append(generate_relative(np.array(lmList)[body_indices], img))

            # Calculate the distance between landmarks 11 and 15 and draw it on the image
            length, img, info = detector.findDistance(lmList[11][0:2],
                                                    lmList[15][0:2],
                                                    img=img,
                                                    color=(255, 0, 0),
                                                    scale=10)

            # Calculate the angle between landmarks 11, 13, and 15 and draw it on the image
            angle, img = detector.findAngle(lmList[11][0:2],
                                            lmList[13][0:2],
                                            lmList[15][0:2],
                                            img=img,
                                            color=(0, 0, 255),
                                            scale=10)

            # Check if the angle is close to 50 degrees with an offset of 10
            isCloseAngle50 = detector.angleCheck(myAngle=angle,
                                                targetAngle=50,
                                                offset=10)

            # Display the frame in a window
            name = f'{directory}/frame_' + str(currentframe) + '.jpg'
            print ('Creating...' + name)

            # writing the extracted images 
            cv2.imwrite(name, img)

            currentframe = currentframe + 1

        # Wait for 1 millisecond between each frame
        cv2.waitKey(1)
    
    output = np.zeros((currentframe, frames[0].shape[0], frames[0].shape[1]))
    output[:] = frames
    print(output.shape) 


video = "video/SixStepPeter.mov"
main(video)