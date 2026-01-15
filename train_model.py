import cv2
import numpy as np
import os
from PIL import Image
import pickle

# Path for face image database
path = 'training'

# Use OpenCV's built-in face detector and recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(path):
    print("Training faces. It will take a few seconds. Wait ...")
    
    imagePaths = []
    # Recursively find all images in the training folder
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                imagePaths.append(os.path.join(root, file))

    faceSamples = []
    ids = []
    names = {} # Map ID numbers to Names
    current_id = 0
    label_map = {} # To keep track of names we've already assigned an ID

    for imagePath in imagePaths:
        # Get the folder name (e.g., "Graciella")
        name = os.path.split(os.path.dirname(imagePath))[-1]
        
        # Assign a number to this name if we haven't seen it yet
        if name not in label_map:
            label_map[name] = current_id
            names[current_id] = name
            current_id += 1
        
        id_num = label_map[name]

        # Convert image to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        # Detect the face inside the image
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id_num)

    # Save the ID-to-Name mapping so the App can use it later
    with open("names.pkl", "wb") as f:
        pickle.dump(names, f)
        
    return faceSamples, ids

# Run the training
faces, ids = getImagesAndLabels(path)

if len(faces) == 0:
    print("Error: No faces found. Make sure your 'training' folder has images!")
else:
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer.yml
    recognizer.write('trainer.yml') 
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting Program")