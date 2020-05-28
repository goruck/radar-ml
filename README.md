*Work in progress*

# radar-ml

The goal of this project is to accurately detect (classify and localize) people, pets and objects using millimeter-wave radar. Self-supervised machine learning techniques, leveraging conventional camera-based object detection, are used to generate radar-based detection models.

Radar-based object detection has certain advantages over camera-based methods, including increased user privacy, lower power consumption, zero-light operation and more flexible placement. However, radar does not have the resolution of most camera-based systems and therefore may be challenged to distinguish objects that have very similar characteristics.

[Vayyar's](https://vayyar.com/) radar in the [Walabot](https://api.walabot.com/) reference design is used in this project. An object detection model running on Google's Coral [Edge TPU](https://coral.ai/) is used to ground truth radar observations in real-time which are in turn used to train an SVM-based model (DNNs are being explored).

## Overview
A photo of the hardware used for this project is shown below. The Walabot radar is mounted up front and horizontally with a camera located at its top center. The white box contains the Google Coral Edge TPU with the camera connected to it over USB and the black box contains a Raspberry Pi 4 with the radar connected to it over USB. The TPU runs a real-time object detection server over [grpc](https://grpc.io/) that a client on the Raspberry Pi communicates with. The object detection server code can be found [here](https://github.com/goruck/detection_server). The Raspberry Pi handles the processing required to determine if a detected radar target is the same as an object detected by the TPU to establish ground truth. The Pi and the TPU communicate over a dedicated Ethernet link to minimize latency which is critical to accurately determining if a radar target is the same as a detected object. The Pi also runs predictions on novel radar targets using a trained SVM model and can be used to train the SVM model from radar targets with ground truth.

![Alt text](./images/training_setup.jpg?raw=true "test setup.")

## Data Collection and Establishing Ground Truth
This section describes how target samples from the radar are collected and how ground truth (i.e, target identification) is established. There may be multiple targets detected by the radar in a single scan so localization is required as part of this process. A key challenge here is accurately translating the 3-D radar image into the 2-D camera image in real-time since people and pets can move rapidly through the scene. Since the radar image is in 3-D, three orthogonal views of the target can be used to generate the dataset. Choosing the view(s) to include in the dataset is a tradeoff exercise between model complexity and size and accuracy.

### Radar and Camera Coordinate Systems
TBA

### Radar Target and Camera Object Correlation
TBA

### Generating and Visualizing Radar Dataset
TBA

## Training the Model
TBA

## Making Predictions
TBA

## Appendix
TBA