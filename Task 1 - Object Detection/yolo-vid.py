"""
        Author : Deepanshu Chauhan
        Object Detection
        #GRIPJANUARY24
        The Sparks Foundation
        IoT and Computer Vision Intern
"""

import numpy as np
import cv2
import argparse

# Parser to specify the input video from the command line.
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

parser.add_argument('-v', '--video-path',
	type=str,   
	help='The path to the video file')

parser.add_argument('-vo', '--video-output-path',
	type=str,
    default='./output.avi',
	help='The path of the output video file')

args = parser.parse_args()

# Load COCO class labels
labelsPath = r'.\yolo-coco\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# Initialize a list of colors to represent each possible class label
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Paths to YOLO weights and model configuration
weightsPath = r'.\yolo-coco\yolov3.weights'
configPath = r'.\yolo-coco\yolov3.cfg'

# Load YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Input Video
cap = cv2.VideoCapture(args.video_path)

# Save the output video
output_video_path = args.video_output_path
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')  # FourCC code for .avi format
fps = 25  
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width, frame_height)

out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

while True:
    ret, image = cap.read()  # Read a frame

    if not ret:
        break

    (H, W) = image.shape[:2]

    # Determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # Construct a blob from the input image and perform a forward
    # pass of the YOLO object detector to get bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    classIDs = []

    # Loop over each of the layer outputs
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Ensure at least one detection exists
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the output frame
    cv2.imshow("Output Video", image)

    # Saving the output Video
    out.write(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()