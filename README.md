
***New: Added support for training Deep Neural Networks and Self-supervised Generative Adversarial Networks***

# radar-ml

Computer vision, leveraging commodity cameras, computing and machine learning algorithms, has revolutionized video tracking, object detection and face recognition. This has created massive value for businesses and their customers as well as some concerns about privacy and control of personal information.

Radar-based recognition and localization of people and things in the home environment has certain advantages over computer vision, including increased user privacy, low power consumption, zero-light operation and more sensor flexible placement. It does suffer some drawbacks as compared to vision systems. For example, radar generally does not have the resolution of most camera-based systems and therefore may be challenged to distinguish objects that have very similar characteristics. Also, although the cost of radar is rapidly declining as automotive and industrial applications grow, it is still generally more expensive than cameras which have long ridden the coattails of mobile phone unit volumes. These factors have slowed its adoption for consumer applications. But more importantly, there are very few radar data sets that can be used to train or fine-tune machine learning models. This is in stark contrast to the ubiquity of computer vision data sets and models.

In this project, you will learn how to accurately detect people, pets and objects using low-power millimeter-wave radar and will see how self-supervised learning, leveraging conventional camera-based object detection, can be used to generate radar-based detection models. Self-supervised learning is autonomous supervised learning. It is a representation learning approach that obviates the need for humans to label data [1].

The steps to gather data using self-supervised learning, train and use a radar-based object detection model are as follows.

- Arrange a camera and a radar sensor to share a common view of the environment.
- Run the radar and a camera-based object detection system to gather information about targets in the environment.
- Create ground truth observations from the radar when it senses targets at the same point in space as the object detector.
- Train a machine learning model such as a Support-Vector Machine (SVM) or a deep neural network (DNN) on these observations and check that it has acceptable accuracy for your application.
- Use the trained machine learning model to predict a novel radar target’s identity.

Vayyar’s radar chip in the Walabot reference design is a great choice to develop these solutions given its flexibility and wide availability. An object detection model running on Google’s Coral Edge Tensor Processing Unit (TPU) is another great choice since it can perform inferences many times faster than the radar can scan for targets.

See companion Medium articles, [Teaching Radar to Understand the Home](https://towardsdatascience.com/teaching-radar-to-understand-the-home-ee78e7e4a0be), [Using Stochastic Gradient Descent to Train Linear Classifiers](https://towardsdatascience.com/using-stochastic-gradient-descent-to-train-linear-classifiers-c80f6aeaff76) and [How to Implement Deep Neural Networks for Radar Image Classification](https://towardsdatascience.com/how-to-implement-deep-neural-networks-for-radar-image-classification-acb1bfcd7f3) for additional information. 

# Training Setup
A photo of the hardware created for this project is shown below. The Walabot radar is mounted up front and horizontally with a camera located at its top center. The white box contains the Google Coral Edge TPU with the camera connected to it over USB and the black box contains a Raspberry Pi 4 with the radar connected to it over another USB.

![Alt text](./images/training_setup.jpg?raw=true "test setup.")

The TPU runs a real-time object detection server over [grpc](https://grpc.io/) that a client on the Raspberry Pi communicates with. The object detection server code can be found [here](https://github.com/goruck/detection_server). The Raspberry Pi handles the processing required to determine if a detected radar target is the same as an object detected by the TPU to establish ground truth. The Pi and the TPU communicate over a dedicated Ethernet link to minimize latency which is critical to accurately determining if a radar target is the same as a detected object. The Pi also runs predictions on novel radar targets using a trained model and is used to fit the model from radar target data with ground truth.


# Radar and Camera Coordinate Systems
The figure below shows the radar (XR, YR, YZ), camera (XC, YC, ZC), image (x, y) and pixel (u, v) coordinate systems. The radar is used as the frame of reference. The camera needs to be placed on the top middle of the radar unit which you can mount either horizontally or vertically (the USB connector is used as a reference, see the Walabot documentation). This ensures that the camera's optical z-axis is aligned with the radar's z-axis and fixed offsets between the camera axis' and the radar axis' can be determined (these are known as the camera extrinsic parameters). Additionally, you should make sure that the camera's angle of view is closely matched to the radar's [Arena](https://api.walabot.com/_features.html#_arena) which is defined in the Python module [common.py](./common.py).

![Alt text](./images/coord_system.jpg?raw=true "coordinate systems.")

Its important for you to understand these relationships to convert a 2-D point from the pixel system (what is actually read from the camera) into the radar's frame of reference. A point in the camera system is shown in blue as P(X, Y, Z) and the corresponding point in the radar system is shown in green as PR(XR, YR, Z). Note that Z is the same in both systems since the radar's z-axis is aligned with the camera's optical z-axis and the image plane is placed as close as possible to radar z = 0.

Note that the camera could be placed anywhere it shares a view of a point with the radar as long as the camera's extrinsic parameters are known so that its view can be rotated and translated into the radar's frame of reference. Placing the camera on the top center of the radar unit as it is done here greatly simplifies the conversion of the coordinate systems but in general having the ability to use arbitrarily placed cameras for self-supervised learning for sensors can be very effective.

You must first calibrate the camera to determine its intrinsic parameters (x and y focal and principal points) and to correct distortion. The intrinsic parameters are used to convert from image to radar coordinates. You can use [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html) to do camera calibration quite easily which was done in this project.

See references [5], [6], [7] and [8] to understand more about camera and world coordinate systems and camera intrinsic and extrinsic parameters.

# Data Collection and Establishing Ground Truth
This section describes how target samples from the radar are collected and how ground truth (i.e, target identification) is established. There may be multiple targets detected by the radar in a single scan so localization is required as part of this process. A key challenge here is accurately translating the 3-D radar image into the 2-D camera image in real-time since people and pets can move rapidly through the scene. Since the radar image is in 3-D, three orthogonal views of the target can be used to generate the dataset. Choosing the view(s) to include in the dataset and the resolution of each is a tradeoff exercise between model complexity, size and accuracy. 

You use the Python module [ground_truth_samples.py](./ground_truth_samples.py) to ground truth radar samples with camera-based observations gathered from the [object detection server](https://github.com/goruck/detection_server). These observations are returned from the server in the form of the centroid coordinates of the detected object's bounding box and its label. The server has about a 20 ms inference latency. This is small in comparison to the 200 ms radar scan rate (which is a function of Arena size and resolution). This 1:10 ratio is designed to minimize tracking error between what the radar senses and what the camera is seeing. The centroid coordinates are converted into the radar frame of reference and then a distance metric is computed between each radar target and the centroid. If the distance is less than a threshold then a match is declared and the radar target's signal and label is stored as an observation.

You can configure [ground_truth_samples.py](./ground_truth_samples.py) to visualize the ground truth process to ensure its working correctly. An example visualization is shown in the screenshot below. You can see in the left window that the centroid of the detected object (which is shown in the right window) is close to the target center. 

![Alt text](./images/realtime_sample_plot.png?raw=true "example visualization.")

# Model Training and Results

## SVM and Logistic Regression
You use the Python module [train.py](./train.py) to train an SVM or Logistic Regression model on the radar samples with ground truth. The samples are scaled to the [0, 1] range, the classes balanced and then the model is fitted using a stratified 5-Folds cross-validator. The fitted model is evaluated for accuracy, pickled and saved to disk.

Three orthogonal views of the radar target's return signal can be obtained. You can think of these as 2-D projections of the 3-D target signal on the radar's X-Y plane, the X-Z plane and the Y-Z plane. Any one of these or any combination of them can be used as an observation. Using all three of them will result in the best accuracy but the dataset (~10k ```float32```'s per sample) and resulting SVM model (~10MB for 1.5k training samples) can be large especially if the radar scan resolution is high. You can think of choosing the optimal combination as a training hyperparameter which is configurable in [train.py](./train.py).

Training results from a run using all projections are shown below.

```text
2020-12-05 05:36:41,951 __main__     INFO     Opening dataset: datasets/radar_samples_25Nov20.pickle
2020-12-05 05:36:41,987 __main__     INFO     Opening dataset: datasets/radar_samples.pickle
2020-12-05 05:36:41,997 __main__     INFO     Maybe filtering classes.
2020-12-05 05:36:41,997 __main__     INFO     Scaling samples.
2020-12-05 05:36:42,029 __main__     INFO     Encoding labels.
2020-12-05 05:36:42,030 __main__     INFO     Found 3 classes and 1137 samples:
2020-12-05 05:36:42,030 __main__     INFO     ...class: 0 "cat" count: 117
2020-12-05 05:36:42,030 __main__     INFO     ...class: 1 "dog" count: 412
2020-12-05 05:36:42,030 __main__     INFO     ...class: 2 "person" count: 608
2020-12-05 05:36:42,030 __main__     INFO     Splitting data set:
2020-12-05 05:36:42,031 __main__     INFO     ...training samples: 909
2020-12-05 05:36:42,031 __main__     INFO     ...validation samples: 114
2020-12-05 05:36:42,031 __main__     INFO     ...test samples: 114
2020-12-05 05:36:42,031 __main__     INFO     Projection mask: ProjMask(xz=True, yz=True, xy=True)
2020-12-05 05:36:42,031 __main__     INFO     Augment epochs: 4
2020-12-05 05:36:42,031 __main__     INFO     Online learning: False
2020-12-05 05:36:42,031 __main__     INFO     Using SVM algo: SGDClassifier.
2020-12-05 05:36:42,031 __main__     INFO     Generating feature vectors.
2020-12-05 05:36:43,405 __main__     INFO     Feature vector length: 10010
2020-12-05 05:36:43,406 __main__     INFO     Balancing classes.
2020-12-05 05:36:43,448 __main__     INFO     Running best fit with new data.
2020-12-05 05:42:19,184 __main__     INFO     
 Best estimator:
2020-12-05 05:42:19,184 __main__     INFO     SGDClassifier(alpha=1e-05, loss='log', n_jobs=-1, penalty='elasticnet',
              random_state=1234, warm_start=True)
2020-12-05 05:42:19,191 __main__     INFO     
 Best score for 5-fold search:
2020-12-05 05:42:19,191 __main__     INFO     0.9710912254536416
2020-12-05 05:42:19,191 __main__     INFO     
 Best hyperparameters:
2020-12-05 05:42:19,192 __main__     INFO     {'alpha': 1e-05, 'average': False, 'l1_ratio': 0.15, 'penalty': 'elasticnet'}
2020-12-05 05:42:19,192 __main__     INFO     Running partial fit with augmented data (epochs: 4).
2020-12-05 05:43:22,791 __main__     INFO     Calibrating classifier.
2020-12-05 05:43:22,811 __main__     INFO     Evaluating final classifier on test set.
2020-12-05 05:43:22,819 __main__     INFO     Accuracy: 0.8859649122807017
2020-12-05 05:43:22,826 __main__     INFO     Confusion matrix:
[[10  3  1]
 [ 1 41  6]
 [ 0  2 50]]
2020-12-05 05:43:23,245 __main__     INFO     Saving confusion matrix plot to: ./train-results/svm_cm.png
2020-12-05 05:43:23,417 __main__     INFO     Classification report:
              precision    recall  f1-score   support

         cat       0.91      0.71      0.80        14
         dog       0.89      0.85      0.87        48
      person       0.88      0.96      0.92        52

    accuracy                           0.89       114
   macro avg       0.89      0.84      0.86       114
weighted avg       0.89      0.89      0.88       114

2020-12-05 05:43:23,418 __main__     INFO     Saving svm model to: ./train-results/svm_radar_classifier.pickle.
2020-12-05 05:43:23,418 __main__     INFO     Saving label encoder to: ./train-results/radar_labels.pickle.
```

The training results for using just the X-Y projection are very similar. There is a relatively minor degradation from the all projection case. However, the model training time and model size (about 10MB vs 1.6MB) are far worse. It does not seem to be worth using all views for training, at least for this particular data set.

The most recently trained Logistic Regression model (which uses all projections) and data set can be found [here](https://drive.google.com/drive/folders/1y8twF6puPXvedXhsFhff45HlMawgHzcS?usp=sharing). This model has the labels ‘person’, ‘dog’ and ‘cat’. You can easily map these names to suit your own household. The model has an input vector of 10,011 features made up of 5,457 features from the Y-Z projection plane, 3,872 features from the X-Z plane and 682 features from the X-Y plane. The number of features in each plane are a function of the radar Arena size and resolution with the default training Arena yielding these numbers.

## Deep Neural Network
Using the same data set you can train a deep neural network (DNN)-based classifier using the Python module [dnn.py](./dnn.py). The DNN model architecture was inspired by Multi-view Convolutional Neural Networks [9]. The data set contains only a few thousand samples (and with known labeling errors) and can only be trained for a small number of epochs before over fitting. Nonetheless you can see that good accuracy is achieved as shown in the training log below.

```text
2021-05-23 06:19:05,223 __main__     INFO     Opening dataset: datasets/radar_samples_25Nov20.pickle
2021-05-23 06:19:05,240 __main__     INFO     Opening dataset: datasets/radar_samples_22Feb21.pickle
2021-05-23 06:19:05,249 __main__     INFO     Opening dataset: datasets/radar_samples_26Feb21.pickle
2021-05-23 06:19:05,260 __main__     INFO     Opening dataset: datasets/radar_samples_28Feb21.pickle
2021-05-23 06:19:05,269 __main__     INFO     Opening dataset: datasets/radar_samples_19Oct20.pickle
2021-05-23 06:19:05,270 __main__     INFO     Opening dataset: datasets/radar_samples_20Sep20.pickle
2021-05-23 06:19:05,302 __main__     INFO     Opening dataset: datasets/radar_samples_20Oct20.pickle
2021-05-23 06:19:05,303 __main__     INFO     Opening dataset: datasets/radar_samples_31Oct20.pickle
2021-05-23 06:19:05,314 __main__     INFO     Using aliased class names.
2021-05-23 06:19:05,315 __main__     INFO     Maybe filtering data set.
2021-05-23 06:19:05,316 __main__     INFO     Scaling samples.
2021-05-23 06:19:05,422 __main__     INFO     Encoding labels.
2021-05-23 06:19:05,424 __main__     INFO     Found 3 classes and 4868 samples:
2021-05-23 06:19:05,424 __main__     INFO     ...class: 0 "cat" count: 450
2021-05-23 06:19:05,424 __main__     INFO     ...class: 1 "dog" count: 1952
2021-05-23 06:19:05,424 __main__     INFO     ...class: 2 "person" count: 2466
2021-05-23 06:19:05,424 __main__     INFO     Class weights: {2: 1.0, 1: 1.26, 0: 5.48}
2021-05-23 06:19:06,525 __main__     INFO     Creating model.
2021-05-23 06:19:07,062 __main__     INFO     Training model.
2021-05-23 06:19:50,582 __main__     INFO     Best loss: 0.5842, Best acc: 85.82%
2021-05-23 06:19:50,583 __main__     INFO     Best val loss: 0.4229, Best val acc: 84.29%
2021-05-23 06:19:50,583 __main__     INFO     Saved best model to ./train-results/dnn
```

## Self-supervised Generative Adversarial Network
As noted above, radar data sets with ground truth labels are not commonplace. Self-supervised learning was used to automatically label observations as outlined above, however that technique occasionally caused labeling errors that negatively affected model training accuracy. Semi-supervised learning is a technique in which both labeled and unlabeled data are used to train a classifier which can obviate the need to label every new radar training sample. In other words, the labeled data gathered so far can be sufficient to train a model on new observations. This technique can be further combined with a Generative Adversarial Network (GAN) to augment an existing data set as is known as an SGAN [10]. You can use the Python module [sgan.py](./sgan.py) to train a Deep Convolutional Network using the SGAN technique from the same data set that was used to train the models above. As you can see from the training log below, decent accuracy is obtained by using only 50 labeled samples per class.

```text
2021-05-16 18:52:02,471 __main__     INFO     Opening dataset: datasets/radar_samples_25Nov20.pickle
2021-05-16 18:52:02,488 __main__     INFO     Opening dataset: datasets/radar_samples_22Feb21.pickle
2021-05-16 18:52:02,496 __main__     INFO     Opening dataset: datasets/radar_samples_26Feb21.pickle
2021-05-16 18:52:02,507 __main__     INFO     Opening dataset: datasets/radar_samples_28Feb21.pickle
2021-05-16 18:52:02,516 __main__     INFO     Opening dataset: datasets/radar_samples_19Oct20.pickle
2021-05-16 18:52:02,517 __main__     INFO     Opening dataset: datasets/radar_samples_20Sep20.pickle
2021-05-16 18:52:02,548 __main__     INFO     Opening dataset: datasets/radar_samples_20Oct20.pickle
2021-05-16 18:52:02,548 __main__     INFO     Opening dataset: datasets/radar_samples_31Oct20.pickle
2021-05-16 18:52:02,559 __main__     INFO     Using aliased class names.
2021-05-16 18:52:02,560 __main__     INFO     Maybe filtering data set.
2021-05-16 18:52:02,561 __main__     INFO     Scaling samples.
2021-05-16 18:52:02,664 __main__     INFO     Encoding labels.
2021-05-16 18:52:02,665 __main__     INFO     Found 3 classes and 4868 samples:
2021-05-16 18:52:02,665 __main__     INFO     ...class: 0 "cat" count: 450
2021-05-16 18:52:02,665 __main__     INFO     ...class: 1 "dog" count: 1952
2021-05-16 18:52:02,665 __main__     INFO     ...class: 2 "person" count: 2466
2021-05-16 18:52:02,665 __main__     INFO     Class weights: {'person': 1.0, 'dog': 1.26, 'cat': 5.48}
2021-05-16 18:52:06,408 __main__     INFO     Starting training loop.
2021-05-16 18:52:06,408 __main__     INFO     n_epochs=15, n_batch=32, 1/2=16, b/e=231, steps=3465
2021-05-16 18:54:18,073 __main__     INFO     Classifier accuracy at step 231: 41.13%
2021-05-16 18:54:18,259 __main__     INFO     Saved: ./train-results/sgan/generated_data_0231.pickle, ./train-results/sgan/g_model_0231.h5, and ./train-results/sgan/c_model_0231.h5
2021-05-16 18:55:06,900 __main__     INFO     Classifier accuracy at step 462: 40.30%
2021-05-16 18:55:07,051 __main__     INFO     Saved: ./train-results/sgan/generated_data_0462.pickle, ./train-results/sgan/g_model_0462.h5, and ./train-results/sgan/c_model_0462.h5
2021-05-16 18:55:55,783 __main__     INFO     Classifier accuracy at step 693: 61.81%
2021-05-16 18:55:55,933 __main__     INFO     Saved: ./train-results/sgan/generated_data_0693.pickle, ./train-results/sgan/g_model_0693.h5, and ./train-results/sgan/c_model_0693.h5
2021-05-16 18:56:44,293 __main__     INFO     Classifier accuracy at step 924: 64.42%
2021-05-16 18:56:44,444 __main__     INFO     Saved: ./train-results/sgan/generated_data_0924.pickle, ./train-results/sgan/g_model_0924.h5, and ./train-results/sgan/c_model_0924.h5
2021-05-16 18:57:33,531 __main__     INFO     Classifier accuracy at step 1155: 73.38%
2021-05-16 18:57:33,724 __main__     INFO     Saved: ./train-results/sgan/generated_data_1155.pickle, ./train-results/sgan/g_model_1155.h5, and ./train-results/sgan/c_model_1155.h5
2021-05-16 18:58:22,376 __main__     INFO     Classifier accuracy at step 1386: 59.88%
2021-05-16 18:58:22,530 __main__     INFO     Saved: ./train-results/sgan/generated_data_1386.pickle, ./train-results/sgan/g_model_1386.h5, and ./train-results/sgan/c_model_1386.h5
2021-05-16 18:59:11,271 __main__     INFO     Classifier accuracy at step 1617: 62.92%
2021-05-16 18:59:11,418 __main__     INFO     Saved: ./train-results/sgan/generated_data_1617.pickle, ./train-results/sgan/g_model_1617.h5, and ./train-results/sgan/c_model_1617.h5
2021-05-16 19:00:00,354 __main__     INFO     Classifier accuracy at step 1848: 63.48%
2021-05-16 19:00:00,501 __main__     INFO     Saved: ./train-results/sgan/generated_data_1848.pickle, ./train-results/sgan/g_model_1848.h5, and ./train-results/sgan/c_model_1848.h5
2021-05-16 19:00:49,175 __main__     INFO     Classifier accuracy at step 2079: 74.53%
2021-05-16 19:00:49,326 __main__     INFO     Saved: ./train-results/sgan/generated_data_2079.pickle, ./train-results/sgan/g_model_2079.h5, and ./train-results/sgan/c_model_2079.h5
2021-05-16 19:01:37,949 __main__     INFO     Classifier accuracy at step 2310: 68.24%
2021-05-16 19:01:38,106 __main__     INFO     Saved: ./train-results/sgan/generated_data_2310.pickle, ./train-results/sgan/g_model_2310.h5, and ./train-results/sgan/c_model_2310.h5
2021-05-16 19:02:26,343 __main__     INFO     Classifier accuracy at step 2541: 69.72%
2021-05-16 19:02:26,493 __main__     INFO     Saved: ./train-results/sgan/generated_data_2541.pickle, ./train-results/sgan/g_model_2541.h5, and ./train-results/sgan/c_model_2541.h5
2021-05-16 19:03:15,088 __main__     INFO     Classifier accuracy at step 2772: 71.92%
2021-05-16 19:03:15,239 __main__     INFO     Saved: ./train-results/sgan/generated_data_2772.pickle, ./train-results/sgan/g_model_2772.h5, and ./train-results/sgan/c_model_2772.h5
2021-05-16 19:04:04,030 __main__     INFO     Classifier accuracy at step 3003: 72.49%
2021-05-16 19:04:04,181 __main__     INFO     Saved: ./train-results/sgan/generated_data_3003.pickle, ./train-results/sgan/g_model_3003.h5, and ./train-results/sgan/c_model_3003.h5
2021-05-16 19:04:52,744 __main__     INFO     Classifier accuracy at step 3234: 73.46%
2021-05-16 19:04:52,894 __main__     INFO     Saved: ./train-results/sgan/generated_data_3234.pickle, ./train-results/sgan/g_model_3234.h5, and ./train-results/sgan/c_model_3234.h5
2021-05-16 19:05:41,900 __main__     INFO     Classifier accuracy at step 3465: 72.51%
2021-05-16 19:05:42,065 __main__     INFO     Saved: ./train-results/sgan/generated_data_3465.pickle, ./train-results/sgan/g_model_3465.h5, and ./train-results/sgan/c_model_3465.h5
```

# Making Predictions
The Python module [predict.py](./predict.py) is used to illustrate how predictions are made on novel radar samples using the trained SVM or Logistic Regression model. You can use a radar prediction Arena different from what was used for training as the predict module will automatically scale the observations as required. However, you should make sure the radar Threshold, Filter Type and Profile are similar to what was used for training. Additionally, the observations need to be constructed from the same orthogonal planes as was used during training.

An example prediction is shown below.

![Alt text](./images/prediction-example.png?raw=true "svm confusion matrix.")

You can use this module and the trained model to run predictions on your own Walabot radar unit.

# License
Everything here is licensed under the [MIT license](./LICENSE).

# Contact
For questions or comments about this project please contact the author goruck (Lindo St. Angel) at {lindostangel} AT {gmail} DOT {com}.

# Acknowledgments
This work was inspired by references [2], [3], and [4].

# References
1. [Self-supervised learning gets us closer to autonomous learning](https://hackernoon.com/self-supervised-learning-gets-us-closer-to-autonomous-learning-be77e6c86b5a)

2. [REAL-TIME HUMAN ACTIVITY RECOGNITION BASED ON RADAR](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi8i9e-h9jpAhXLup4KHRsdCNMQFjABegQIBhAB&url=http%3A%2F%2Fcardinalscholar.bsu.edu%2Fbitstream%2Fhandle%2F123456789%2F201689%2FGuoH_2019-2_BODY.pdf%3Fsequence%3D1&usg=AOvVaw2Ps7ptIYpZGBYPnMQyPArp)

3. [Through-Wall Pose Imaging in Real-Time with a Many-to-Many
Encoder/Decoder Paradigm](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi_19iHiNjpAhXIi54KHUy1A_4QFjABegQIBxAB&url=https%3A%2F%2Farxiv.org%2Fpdf%2F1904.00739&usg=AOvVaw05g69S1Wbh3JwGiKmQPeH8)

4. [Unobtrusive Activity Recognition and Position Estimation for Work Surfaces Using RF-Radar Sensing](https://dl.acm.org/doi/fullHtml/10.1145/3241383)

5. [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html)

6. [Camera calibration](http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT9/node2.html)

7. [Geometric Camera Parameters](https://www.cse.unr.edu/~bebis/CS791E/Notes/CameraParameters.pdf)

8. [To calculate world coordinates from screen coordinates with OpenCV](https://stackoverflow.com/questions/12007775/to-calculate-world-coordinates-from-screen-coordinates-with-opencv)

9. [Multi-view Convolutional Neural Networks for 3D Shape Recognition](https://arxiv.org/abs/1505.00880)

10. [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/pdf/1606.01583.pdf)