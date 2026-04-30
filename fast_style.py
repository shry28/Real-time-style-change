import cv2
import numpy as np
import os

# List of all the models we downloaded
models = ["candy.t7", "mosaic.t7", "starry_night.t7", "udnie.t7"]
current_idx = 0

# Load the first model
print(f"Loading initial model: {models[current_idx]}...")
net = cv2.dnn.readNetFromTorch(models[current_idx])

# Start webcam
cap = cv2.VideoCapture(0)
print("Webcam started.")
print("--> Press 'n' to cycle to the next style.")
print("--> Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Preprocess and forward pass
    blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    # Post-process the output
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output /= 255.0
    output = output.transpose(1, 2, 0)
    output = np.clip(output, 0.0, 1.0)
    
    output_image = (output * 255).astype(np.uint8)

    # --- THE FIX: Make the array contiguous in memory for OpenCV ---
    output_image = np.ascontiguousarray(output_image)

    # Add text showing the current style
    style_name = models[current_idx].replace('.t7', '').replace('_', ' ').title()
    cv2.putText(output_image, f"Current Style: {style_name}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the styled output
    cv2.imshow("Multi-Style Real-Time Transfer", output_image)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        # Move to the next style in the list, loop back to 0 if at the end
        current_idx = (current_idx + 1) % len(models)
        print(f"Switching to {models[current_idx]}...")
        # Load the new model into memory
        net = cv2.dnn.readNetFromTorch(models[current_idx])

cap.release()
cv2.destroyAllWindows()