*Work in progress*

# radar-ml

The goal of this project is to accurately detect (classify and localize) people, pets and objects using millimeter-wave radar. Self-supervised machine learning techniques, leveraging conventional camera-based object detection, are used to generate radar-based detection models.

Radar-based object detection has certain advantages over camera-based methods, including increased user privacy, lower power consumption, zero-light operation and more flexible placement. However, radar does not have the resolution of most camera-based systems and therefore may be challenged to distinguish objects that have very similar characteristics.

[Vayyar's](https://vayyar.com/) radar in the [Walabot](https://api.walabot.com/) reference design is used in this project.

An object detection model running on Google's Coral [Edge TPU](https://coral.ai/) is used to ground truth radar observations in real-time during training. 

## Test Setup
A photo of the hardware used for the self-supervised learning and predictions is shown below. The Walabot radar is mounted up front and horizontally with a camera located at its top center. The camera is connected to the Google Edge TPU which is in the white box to perform real-time object detection. The radar is connected to a Raspberry Pi 4 which handles the processing required to determine a detected radar target is the same as an object detected by the TPU. The Raspberry Pi and the TPU communicate over a dedicated Ethernet link to minimize latency. 

![Alt text](./images/training_setup.jpg?raw=true "training setup.")

The Google TPU runs an object detection server over grpc that a client on the Raspberry Pi communicates with. The object detection server code can be found [here]().

## Training
TBA

## Predictions
TBA
