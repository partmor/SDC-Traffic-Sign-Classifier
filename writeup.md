# **Traffic Sign Recognition** 

## Writeup
---


[//]: # (Image References)

[histograms]: ./examples/histograms.png
[set30]: ./examples/set30.png
[original_sample]: ./examples/original_sample.png 
[prep_sample]: ./examples/prep_sample.png 
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

### Data Set Summary & Exploration

The [given German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) consists of a total of 51839 32x32 px color images (3-channel RGB), already split into the training, validation, and test sets:

| Data set			    |     Number of samples | 
|:---------------------:|:---------------------:| 
| Training      		| 34799					| 
| Validation     		| 4410 					|
| Test					| 12630					|

The dataset contains samples of 43 types of traffic signs, listed [here](signnames.csv).

![histograms]

Certain classes are much more abundant than others: the datasets are highly unbalanced. However, the given train, validation and test sets present a similar composition regarding the proportion of samples for each class. The latter is an important fact to bear in mind when assessing a classifier in a scenario that inherently involves rare classes: validation and test sets should preserve the percentage of samples for each class for the validation/test scores to be meaningful. Indeed, the given sets seem to have been split in a stratified manner.

It is also worth highlighting that the images in the dataset have been extracted from video-tracks. Hence, groups of samples represent the same "real world instance", but captured under different conditions (distance, angle, speed, etc.) - corresponding to a changing position of the vehicle with respect to the sign: 

![set30]

This is a reminder that sample shuffling is crucial during training. 

A final fact to highlight is that lighting conditions and integrity of the images vary hugely throughout the dataset: images can be dark, blurry due to vehicle movement, or partially ocluded by surrounding objects.

### Data Pipeline and Model Architecture

The preprocessing pipeline of the data consists of the following steps;
+ Conversion to **grayscale**: research by [Sermanet & LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggests that models using luminance solely (Y channel) outperform those using RGB channels.
+ [0, 1] **normalization**: convenient for convergence of the gradient descent algorithm in the training phase.
+ Per-sample **mean substracion**: this normalization has the property of removing the average brightness (intensity) of the data point. In this case, we are not interested in the illumination conditions of the image, but more so in the content.

For instance, the following samples:

![original_sample]

are transformed into:

![prep_sample]
