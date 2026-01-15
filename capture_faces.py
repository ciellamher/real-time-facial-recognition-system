import cv2
import os

# CONFIGURATION
name = "Graciella"  # Change this to the person's name
save_folder = f"training/{name}"

# Create the folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Created folder: {save_folder}")

# Start the Camera (0 is usually the default Mac webcam)
cam = cv2.VideoCapture(0)

count = 0
print("Press 's' to save a photo. Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Press 's' to Save", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        # Save the image
        img_name = f"{save_folder}/img_{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} saved!")
        count += 1
    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()