# importing libraries 
import os 
import cv2
import numpy as np
from PIL import Image


# Video Generating function 
def generate_video(project_name): 
    image_folder = f'./frames/{project_name}/' # make sure to use your folder 
    video_name = f'{project_name}_out.mp4'
      
    images = [img for img in os.listdir(image_folder) 
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")] 
     
    # Array images should only consider 
    # the image files ignoring others if any

    numbers = []

    for img in images:
        numbers.append(int(img.split(".jpg")[0].split("_")[-1]))

    images = np.array(images)
    images = images[np.argsort(numbers)]
    images = list(images)
  
    frame = cv2.imread(os.path.join(image_folder, images[0])) 
  
    # setting the frame width, height width 
    # the width, height of first image 
    height, width, layers = frame.shape   
  
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 32, (width, height))  
  
    # Appending the images to the video one by one 
    for image in images:  
        video.write(cv2.imread(os.path.join(image_folder, image)))  
      
    # Deallocating memories taken for window creation 
    cv2.destroyAllWindows()  
    video.release()  # releasing the video generated 
  
  
# Calling the generate_video function 
generate_video("SixStepPeter") 