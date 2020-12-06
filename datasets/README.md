# data set folder
This folder contains labeled radar data sets gathered by ```ground_truth_samples.py``` for use in training the radar classifier. 

All data sets except for the most recent are postfixed with the date of the most recent sample in the data set.

The data sets with date 25 Nov 20 and later used the object detection model ```ssd_mobiledet_coco_edgetpu``` running on the object detection server. This model and the data sets labled by it are the most accurate and should be used for training new classifiers. 

A data set is a pickled Python dict of the form:
```python
{'samples': samples, 'labels': labels}
```

```samples``` is a list of N radar projection numpy.array tuple samples in the form:
```python
[(xz_0, yz_0, xy_0), (xz_1, yz_1, xy_1),...,(xz_N, yz_N, xy_N)]
```
```labels``` is a list of N numpy.array class labels corresponding to each radar projection sample of the form:
```python
[class_label_0, class_label_1,...,class_label_N]
```

This data was catured in my house in various locations designed to maximize the variation in detected objects (currently only people, dogs and cats), distance and angle from the radar sensor.

The data set contains known labeling errors mostly stemming from the object detector mistaking my cat for my dog which happens (subjectively) at about a 10% error rate. A future effort will attempt to fine-tune the object detector to reduce the error. This error will get propagated to the radar classifier trained from this data set. 