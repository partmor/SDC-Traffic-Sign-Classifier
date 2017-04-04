# **Traffic Sign Recognition** 

## Writeup
---


[//]: # (Image References)

[histograms]: ./examples/histograms.png
[set30]: ./examples/set30.png
[original_sample]: ./examples/original_sample.png 
[prep_sample]: ./examples/prep_sample.png 
[architecture]: ./examples/architecture.jpg

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

###  Data preprocessing

The preprocessing pipeline of the data consists of the following steps;
+ Conversion to **grayscale**: [Sermanet et al.](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggest that models using luminance solely (Y channel) outperform those using RGB channels.
+ [0, 1] **normalization**: convenient for convergence of the gradient descent algorithm in the training phase.
+ Per-sample **mean substracion**: this normalization has the property of removing the average brightness (intensity) of the data point. In this case, we are not interested in the illumination conditions of the image, but more so in the content.

For instance, the following samples:

![original_sample]

are transformed into:

![prep_sample]

### Model Architecture

For this challenge I use a **2-stage ConvNet** with **multi-scale** feature architecture (MS) followed by a classifier that consists of **two fully connected layers**. 

![architecture]

Usual ConvNets are organized in strict feed-forward layered architectures in which the output of one layer is fed only to the layer above (single-scale feature architecture). Instead, in a multi-scale architecture, the output of the first stage is branched out and fed to the classifier, in addition to the output of the second stage. Additionally, a second subsampling stage is applied on the branched output, yielding higher accuracies than with just one. 

The motivation for combining representation from multiple stages in the classifier is to provide different scales of receptive fields to the classifier. The second stage extracts global and invariant shapes and structures, while the first
stage extracts local motifs with more precise details.

A summary of the parameters used for the model can be found in the following table:

| Stage    |     Layer | Description | Output size |
|:---------------------:|:---------------------:| :---------------------:|:---------------------:|
| **Input**      		| 	Input		| grayscaled image (Y channel only) | 32x32x1|
| **1st stage ConvNet (CN1)**      		| 	Convolution		| 30 5x5 filters; 1x1 stride; valid padding | 28x28x30 |
|       		|  	ReLU		| | 28x28x30
| 				| Max Pooling	|2x2 stride; valid padding| 14x14x30
| **2nd stage ConvNet (CN2)**      		| 	Convolution		| 60 5x5 filters; 1x1 stride; valid padding | 10x10x60 |
|       		|  	ReLU		| | 10x10x60 |
| 				| Max Pooling	|2x2 stride; valid padding| 5x5x60 |
| **CN1 branch**      		| 	Max Pooling		| Pooling applied on CN1 output; 4x4 stride; valid padding | 3x3x30
| **Preparation for classifier**		|  	Flattening and concatenation		| Branched CN1 output and CN2 output are flattened and concatenated | 1x1770
| **1st stage FC (FC1)**				| Fully connected layer	|100 hidden units| 1x100
| 	| ReLU	| | 1x100
| **2nd stage FC (FC2)**				| Fully connected layer	|100 hidden units| 1x100
| 	| ReLU	| | 1x100
| **Output**				| Fully connected layer	|43 units| 1x43
| 	| Softmax	| | 1x43

This multi-scale architecture was chosen since it quickly outperformes simpler single-scale architectures like LeNet-5: naive tuning of the MS model could easilly outperform a more carefully tuned LeNet-5 by 4%. The traffic sign classification problem is inherently more complex than the handwritten digit classification, and MS gives the classifier the ability to use low-level features together with high-level ones.

### Model training

In the training process the objective function to minimize is the cross entropy loss over the training set, monitoring the evolution of training and validation accuracy to ensure overfitting is avoided. Training cross entropy is minimized using the **Adam** optimizer, a method that computes adaptive learning rates for each parameter. [Kingma et al., 2014](https://arxiv.org/abs/1412.6980), show empirically that Adam compares favorably to other adaptive learning-method algorithms.

Overfitting is mitigated via:
+ Dropout: applied after CN1, CN2, FC1 and FC2, with ascending values towards the output, as suggested by [Srivastava et al., 2014](https://www.google.es/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwi-p5Ci34vTAhUEPBQKHUnMDegQFggcMAA&url=https%3A%2F%2Fwww.cs.toronto.edu%2F~hinton%2Fabsps%2FJMLRdropout.pdf&usg=AFQjCNFModVeeXkqtxn_TXeKPB0zFtw5ew&sig2=NAInnEhv1iJyKfk5yx87Sw). Ascending dropout values towards the output, more aggressive on the fully connected layers, have proven to work better: [10, 20, 50, 50] % respectively.
+ L2 regularization:  L2 penaltization is applied only on weights of the fully connected layers. A scale of 1e-4 yielded best performance.

In order to reduce the number of explicit hyperparameters, CN and FC weights are initialized using the Xavier initializer. This initializer is designed to keep the scale of the gradients roughly the same in all layers. In uniform distribution this ends up being the range.

Using a learning rate of 5e-4, a batch size of 128, and 100 epochs with the given hyperparameter setup yielded a *winning* model between epochs 70 and 80:

| Dataset			    |     Accuracy | 
|:---------------------:|:---------------------:| 
| Training      		| 0.982				| 
| Validation     		| 0.991 					|
| Test					| 0.964				|

