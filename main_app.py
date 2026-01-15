import cv2
import pickle

# 1. Load the trained model and the names dictionary
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read('trainer.yml')
    labels = pickle.loads(open("names.pkl", "rb").read())
except FileNotFoundError:
    print("Error: Missing 'trainer.yml' or 'names.pkl'. Did you run train_model.py?")
    exit()

# 2. Load the face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

# 3. Start the Camera
cap = cv2.VideoCapture(0) # 0 is the default Mac webcam

print("Starting Camera... Press 'q' to quit.")

while True:
    ret, img = cap.read()
    if not ret: break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the live video
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
    )

    for(x,y,w,h) in faces:
        # Draw a box around the face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        # Predict who this is based on your training
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is less than 100, it is a match 
        if (confidence < 100):
            name = labels[id]
            confidence_text = f"  {round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"  {round(100 - confidence)}%"
        
        # Display the name on screen
        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_text), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    cv2.imshow('Final Face App', img) 

    # Press 'q' to quit the app
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()