import numpy as np
import tensorflow as tf
from guizero import App, Box, Picture, Text, PushButton
import cv2
import subprocess
from io import BytesIO
from tkinter import PhotoImage
from PIL import Image
import time

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mapping of gesture predictions
GESTURE_MAP = {
    0: "Rock",
    1: "Paper",
    2: "Scissors"
}

# Initialize scores
score = {"Wins": 0, "Losses": 0, "Ties": 0}

# Helper function to preprocess the image
def preprocess_image(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# Function to make predictions
def predict_gesture(frame):
    preprocessed_image = preprocess_image(frame)
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_index = np.argmax(output_data)

    print("Model Output Probabilities:", output_data)
    print("Predicted Index:", prediction_index)

    return GESTURE_MAP.get(prediction_index, "Unknown")

# Initialize GUI with guizero
app = App("Rock, Paper, Scissors", width=640, height=480)
box = Box(app, width="fill", height="fill")
camera_picture = Picture(box, width=320, height=240)
result_text = Text(app, text="Make your gesture!", size=20)
score_text = Text(app, text="Wins: 0  Losses: 0  Ties: 0", size=16)
pi_gesture_display = Text(app, text="", size=18)

# Function to capture an image from the camera
def capture_image():
    subprocess.run([
        "libcamera-still", "--output", "frame.jpg",
        "--width", "640", "--height", "480", "--timeout", "1",
        "--nopreview"
    ])

    frame = cv2.imread("frame.jpg")
    if frame is not None:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame

# Function to update the camera feed in the GUI
def update_camera_feed():
    frame = capture_image()

    if frame is not None:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (320, 240))

        image_bytes = BytesIO()
        image = Image.fromarray(frame_resized)
        image.save(image_bytes, format="PNG")
        image_bytes.seek(0)
        photo = PhotoImage(data=image_bytes.read())

        camera_picture.image = photo
        camera_picture.tk.update()

    app.after(25, update_camera_feed)

# Animation function for Pi's gesture
def animate_pi_gesture():
    gestures = ["Rock", "Paper", "Scissors"]
    for _ in range(15):  # Loop through gestures 10 times
        pi_gesture_display.value = f"Pi: {np.random.choice(gestures)}"
        time.sleep(0.1)  # Add a short delay to simulate animation
        app.update()

# Function to start the game
def start_game():
    global score
    frame = capture_image()
    if frame is not None:
        # Player's choice
        player_choice = predict_gesture(frame)
        result_text.value = f"You chose: {player_choice}"

        # Simulate Pi's gesture animation
        animate_pi_gesture()

        # Pi's final random choice
        pi_choice = np.random.choice(["Rock", "Paper", "Scissors"])
        pi_gesture_display.value = f"The Pi Chose: {pi_choice}"

        # Determine winner
        if player_choice == pi_choice:
            result_text.append("\nIt's a tie!")
            score["Ties"] += 1
        elif (player_choice == "Rock" and pi_choice == "Scissors") or \
             (player_choice == "Scissors" and pi_choice == "Paper") or \
             (player_choice == "Paper" and pi_choice == "Rock"):
            result_text.append("\nYou win!")
            score["Wins"] += 1
        else:
            result_text.append("\nYou lose!")
            score["Losses"] += 1

        # Update the score display
        score_text.value = f"Wins: {score['Wins']}  Losses: {score['Losses']}  Ties: {score['Ties']}"

# Add button to start game
play_button = PushButton(app, text="Play", command=start_game, width=10, height=2)

# Start the camera feed update loop
update_camera_feed()

# Run the GUI
app.display()
