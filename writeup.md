# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/histogram.png "Histogram of Training Dataset"
[image2]: ./writeup_images/sample_images.png "Randomly Selected Images"
[image3]: ./writeup_images/8-878.png "Before Histogram Equalization"
[image4]: ./writeup_images/8-878-equalized.png "After Histogram Equalization"
[image5]: ./writeup_images/18.jpg "Traffic Sign 18"
[image6]: ./writeup_images/22.jpg "Traffic Sign 22"
[image7]: ./writeup_images/3.jpg "Traffic Sign 3"
[image8]: ./writeup_images/34.jpg "Traffic Sign 34"
[image9]: ./writeup_images/14.jpg "Traffic Sign 14"
[image10]: ./writeup_images/35.jpg "Traffic Sign 35"
[image11]: ./writeup_images/8.jpg "Traffic Sign 8"
[image12]: ./writeup_images/36.jpg "Traffic Sign 36"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](hhttps://github.com/jrkwon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. The figure below is a histogram showing how the training dataset is organized. The sample size of some images is much smaller than others. We may need to augment the dataset to have evenly distributed labeled dataset. 

![alt text][image1]

The images below are randomly selected from the training dataset one from each label to explore how the training dataset looks like. As the images show, the levels of brightness of the images quite vary.

![alt text][image2]

### Design and Test a Model Architecture

As the first step, I decided to normalize the brightness of images since I found out that it is quite different from image to image. The technique that I used is the histogram equalization. 

Here is an example of a traffic sign image before and after equalizing histograms.

![alt text][image3]![alt text][image4]

As the last step, I normalized the image data because it is convenient to have pixels values between -1 and 1 instead of between 0 and 255.

I decided not to generate additional data because I had a good accuracy from my convolutional network architecture after tweaking the hyperparameters and the number of layers. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
	| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64      	 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Fully connected		|  inputs 1600 (5x5x64), output 512, dropout (keep_prob: 0.7)       									|
| RELU					|												|
| Fully connected		|  inputs 512, output 128, dropout (keep_prob: 0.7)       									|
| RELU					|												|
| Fully connected		|  inputs 128, output 43 (n_classes)       									|
| RELU				|        									|
 | Softmax				|        									|
 
To train the model, I used the Adam optimizer with the hyperparameters as follows. 

* The number of epochs: 20
* The batch size: 100
* The learning rate: 0.001 

My final model results were:

* training set accuracy: 97.1%
* validation set accuracy: 95.2%
* test set accuracy: 93.8%

I chose an iterative approach. The first architecture that I tried was LeNet that happens to have its input size is 32x32 that is the same size with the traffic sign dataset except for the number of channels. This initial architecture has never had its accuracy 90%. I tried to convert the original three channel images to the grayscale ones. The accuracy was not improved much. The next change was to use 32 or 64 for the filter depth which is the higher than the original LeNet. This also did not improve the accuracy much. These approaches made me think to use different network architecture from the LeNet. The new one had one convolutional network that is followed by three fully connected layers. With no satisfactory accuracy, I tried many different hyperparameters with different filter sizes. They gave me no more than 90% accuracy. So I jumped to another network architecture. I added one more convolutional layer. And 30% dropout was added to the first and second fully connected layers. The change made the accuracy near 93%, but it was not enough. Again, I tried to use different sizes of input and output in the fully connected layers. But the accuracy did not go up. After intense contemplation, I found out that my dropout rate was being applied to the evaluation stage, which is not a right thing to do. I modified the value as TensorFlow's placeholder and gave 1.0 for evaluation while the 20% dropout rate was still used in the training stage. This final touch made the model get more than 93% accuracy with the test dataset.

### Test a Model on New Images

Here are eight German traffic signs that I found on the web:

![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]
 ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12] 

The second image might be difficult to classify the sign because the bumpy road sign is small. The fourth image might also be difficult to classify the sign since it has a large red circle on the sign. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General caution     		| General caution   									| 
| Bumpy road     			| Bumpy road 										|
| Speed limit (60km/h)					| Speed limit (60km/h)											|
| Turn left ahead	      		| Turn left ahead					 				|
| Stop			| Stop      							|
| Ahead only			| Ahead only      							|
| Speed limit (120km/h)			| Speed limit (120km/h)      							|
| Go straight or right			| Go straight or right      							|


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.8%.

The code for making predictions on my final model is located in the 10th cell of the Ipython notebook.

For the first image, the model is quite sure that this is a stop sign (probability of 1.0), and the image does contain a general caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| General caution   									| 
| 5.53e-31     				| Pedestrians 										|
| 1.17e-33					| Traffic signals 										|
| 9.53e-38	      			| Speed limit (30km/h)				 				|
| 0.0				    | Speed limit (20km/h))      							|


For the second image, the model is quite sure that this is a bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Bumpy road  									| 
| 0.0012     				| Bicycles crossing 										|
| 8.55e-05					| No vehicles 											|
| 2.83e-05	      			| Children crossing				 				|
| 1.997e-05				    | Speed limit (70km/h)    							|


For the third image, the model is quite sure that this is a speed limit (60km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Speed limit (60km/h)  									| 
| 0.0012     				| Speed limit (80km/h) 										|
| 8.55e-05					| End of no passing 											|
| 2.83e-05	      			| Speed limit (50km/h)				 				|
| 1.997e-05				    | No vehicles     							|

For the fifth image, the model is again quite sure that this is a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop 									| 
| 4.73e-16     				| Keep left 										|
| 4.43e-16					| No entry										|
| 1.24e-16	      			| Traffic signals			 				|
| 4.58e-17				    |  Speed limit (70km/h)							|

For the rest of images, the model is almost certain about the predictions since the probabilities of the first choice are almost 1.0.
