import cv2
import numpy as np
import time
import pygame
import tensorflow as tf
from keras.layers import TFSMLayer  # Updated for Keras 3

# Load the SavedModel using TFSMLayer
model = TFSMLayer("model.savedmodel", call_endpoint="serving_default")

# Load the class labels
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize Pygame for alarm sound
pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

# Start webcam capture
cap = cv2.VideoCapture(0)

# Variables to track time when eyes are closed
closed_start_time = None
ALERT_THRESHOLD = 2  # Seconds eyes need to be closed to trigger alarm

# Set threshold for sleepiness detection (75%)
SLEEPY_THRESHOLD = 0.75

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image for the model
    image = cv2.resize(frame, (224, 224))
    input_data = np.expand_dims(image, axis=0).astype(np.float32)
    input_data = (input_data / 127.5) - 1  # Normalize image

    # Run the inference
    prediction = model(input_data)

    # Debugging: Print the entire prediction structure
    print("Prediction Structure:", prediction)

    # Get the prediction values and class names
    prediction_values = prediction['sequential_3'].numpy().flatten()
    print("Prediction Values:", prediction_values)

    # Get the predicted class and its confidence
    index = np.argmax(prediction_values)
    class_name = class_names[index]
    confidence = prediction_values[index]

    # Display prediction on the screen
    label = f"{class_name}: {confidence * 100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Drowsiness Detection", frame)

    # Debugging: Check if "sleepy" is predicted with sufficient confidence
    print(f"Class Name: {class_name}, Confidence: {confidence * 100:.2f}%")

    # Check if the "sleepy" class is detected and confidence is high enough
    if "sleepy" in class_name.lower() and confidence >= SLEEPY_THRESHOLD:
        print(f"Sleepy detected with {confidence * 100:.2f}% confidence")

        # If it's the first time detecting "sleepy", start the timer
        if closed_start_time is None:
            closed_start_time = time.time()
            print("Starting timer for sleepy...")

        # Check if eyes have been closed for more than the threshold
        elif time.time() - closed_start_time >= ALERT_THRESHOLD:
            print("Eyes closed for more than 2 seconds. Triggering alarm!")
            if not pygame.mixer.music.get_busy():  # Prevent repeated alarms
                pygame.mixer.music.play()

    else:
        # Reset timer if eyes are not "sleepy"
        closed_start_time = None
        pygame.mixer.music.stop()

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and clean up
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()