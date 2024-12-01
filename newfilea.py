from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (replace 'yolov8n.pt' with a larger model for higher accuracy if needed)
model = YOLO('yolov8n.pt')  # Options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.

# Access the camera feed
cap = cv2.VideoCapture(0)  # Replace 0 with your camera index if needed

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Perform object detection
    results = model(frame)

    # Annotate the frame with the detection results
    annotated_frame = results[0].plot()  # Plot the predictions on the frame

    # Display the annotated frame
    cv2.imshow('YOLOv8 Object Detection', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
