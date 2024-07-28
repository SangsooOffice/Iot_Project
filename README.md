# Project Title
Building a Self-Driving Car Using Raspberry Pi. This project was conducted during the IoT System Design and Practice class in the first semester of 2023.


# Project Process

**Data Collection**
- Data were collected while driving the car directly on the track.

**Preprocessing**
- Image cropping to show only the track.
- Converting images with only track lines to gray scale
- Apply Gaussian Blur to reduce noise.
- Apply morphology operations to detect lines well.
- Binary to separate white lines and background

<img src="https://github.com/user-attachments/assets/0cc2fc33-b96f-46d4-9bd3-82bc760eb904" alt="After Modeling Result" style="display: inline; margin-left: 10px;"/>


**Modeling**
- Define Nvidia model consisting of multiple convolution layers and fully connected layers, and progress the training.
- After training, convert to TensorFlow Lite format for model weight reduction

# Object Detection
- Load SSD MobileNet V2 COCO model to perform object detection. 
- Select only objects with reliability of 0.5 or higher among the objects.
- The car stops if it is a specific object.

# Difficult Point
- Thinking of preprocessing to recognize lines well.
- Thinking of techniques to lighten the model.