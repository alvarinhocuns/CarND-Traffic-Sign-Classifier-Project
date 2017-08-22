# **Traffic Sign Recognition** 

## Álvaro Dosil Suárez

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/step1-exploratory.jpg "Extrapolation"
[image2]: ./results/step1-randomImages.png "Random images"
[image3]: ./results/step1-gray.png "Gray scale"
[image4]: ./results/step2-PrecisionRecall.png "Precision and Recall"
[image5]: ./results/step1-step1-randomAugmented.png "Augmenting"
[image6]: ./CarND-Traffic-Sign-Classifier-Project/traffic-sings-data/forbiddendir.ppm "Signal1"
[image7]: ./CarND-Traffic-Sign-Classifier-Project/traffic-sings-data/max50.ppm "Signal2"
[image8]: ./CarND-Traffic-Sign-Classifier-Project/traffic-sings-data/pass.ppm "Signal3"
[image9]: ./CarND-Traffic-Sign-Classifier-Project/traffic-sings-data/roundabout.ppm "Signal4"
[image10]: ./CarND-Traffic-Sign-Classifier-Project/traffic-sings-data/stop.ppm "Signal5"
[image11]: ./CarND-Traffic-Sign-Classifier-Project/traffic-sings-data/works.ppm "Signal6"
[image12]: ./results/step5-Performance.png "Performance"
[image13]: ./results/step4-image.png "Truck"
[image14]: ./results/step4-conv1.png "conv1"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the dataset. It is a bar histogram showing how the number of events per class and per each of the data sets.

![alt text][image1]

Furthermore, I plotted some random images from the training dataset to have an idea of how they look like.

![alt text][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The preprocessing steps that I did were:
 * Gray scale: Even though naively at first moment I thought that converting to gray scale would decrease the performance of the model since I was discarding information, the final result is that gray scale pictures result in around 2% higher accuracy than RGB pictures. Below is an example of an image before and after grayscaling.
![alt text][image3]
 
 * Normalize the data: The goal is to obtain a common data sample range with mean=0 and stddev=1. The output of this step are black images so I am not uploading them.

 * Augment: Since after including two dropout layers to the model it still over fitted, I decided to augment the data. I took the pictures belonging to those classes with lower precision and recall and applied to each of them three different transformations:
 ** Translate the picture a random number of pixels between -3 and 3, in x and y axis.
 ** Rotate the picture a random angle between -3 and 3 degrees.
 ** Zoom the picture between 1 and 1.2 times its original size.
 The number of augmented classes were 19 and I created 8580 pictures more that resulted in a final training set of 43379 pictures.
 
 In the next figure are shown some random augmented figures:
![alt text][image5]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         	|     Description				| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 32x32x1 Gray scale image   			| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			|
| Fully connected	| Input=400, output=120        			|
| RELU			|						|
| Dropout		| Keep prob=0.5					|
| Fully connected	| Input=120, output=84        			|
| RELU			|						|
| Dropout		| Keep prob=0.5					|
| Fully connected	| Input=84, output=43        			|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer with a learning rate of 0.001. The batch size used was 128 and 20 epochs. The keep probability of the two dropout layers was chosen to be 0.5 in each of them.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.2%
* validation set accuracy of 95.1%
* test set accuracy of 93.1%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  The first architecture that I used was the LeNet-5 used in the last project of the course. I chose this architecture because some of the recommended papers for the project used it and they got very good results
  
* What were some problems with the initial architecture?
  The only issue that I found is that the architecture over fitted, so I had to include more layers.
  
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  At first moment, the architecture over fitted, so I decided to include one dropout step at the beginning, and finally two dropout layers. The first one is located between layer3 and layer4, and the second one between layer4 and the output layer.
  
* Which parameters were tuned? How were they adjusted and why?
  The parameters tuned were:
  ** Keep probability: Ranging from 0.4 and 0.5. The final result was 0.5
  ** Learning rate: Ranging from 0.0005 and 0.002. The final result was 0.001
  ** Number of epochs: The final result was 20.
  
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  I decided to use a convolutional architecture because they were used many times to recognize patterns with large variability, such as handwritten characters. So I would expect them to work better in this case.
  The dropout later helps to fit the general LeNet-5 architecture to our sample.
  
If a well known architecture was chosen:
* What architecture was chosen?
The architecture chosen was the LeNet-5.

* Why did you believe it would be relevant to the traffic sign application?
As I already commented, the convolutional layers are specifically designed to recognize patterns and the LeNet architecture was successfully implemented to recognize handwritten characters. Since the traffic signs present a much lower variability than the handwritten characters, and each type send a very clear information, I think that they can be successfully used also to this problem.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The model was trained using the training sample, and the accuracy obtained is high. Since the same model applied to a different data set, in this case the validation dataset, is also high, we can assume that the model is working well. Nevertheless, since we are trying to optimize the accuracy in the validation dataset and we are modifying the model to reach this goal, it is possible that it can learn also about the validation dataset. Because of this, we apply the model also to a third dataset, the test set. Since the accuracy is also very high in this third sample we can assume that it is working properly.




###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10] ![alt text][image11]

All these images are quite good compared with the ones from the dataset used in the project. Nevertheless, the roundabout sign could be difficult to classify since it is a little bit stretched and it has a water mark than can confuse the model. Also, the speed limit signal could present some difficulties to recognize the borders and the 5 can be misidentified as a 3 or a 6 easily.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        	| 
|:---------------------:|:---------------------------------------------:|
| No entry			| No entry				|
| Speed limit 50		| Speed limit 30			|
| No overtaking			| No overtaking				|
| Roundabout			| Roundabout				|
| Stop				| Stop   				|
| Works				| Works					|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This is lower than the results obtained in the validation and test sets, but since we have only 6 pictures our statistical uncertainty is compatible with the validation and test results.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The next image shows the highest 5 probabilities for each image class in semilogarithmic scale. Classes 17, 9, 14 and 25 are predicted with a very high probability, being the second highest probability class much lower than the correct one.
Class 40 (Roundabout) is predicted with a high accuracy, but the second highest class (12-priority road) presents a quite high probability compared with the other results, so I would expect to find some bad predictions in a larger sample.
Finally, class 2 (speed limit 50 km/h) is predicted as class 1 (speed limit 30 km/h) and the other probabilities are also quite high. It is remarkable that the correct class isn't among the 5 highest probabilities despite of being a quite good image. This case should be studied and corrected.
![alt text][image12]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The next pictures represent the initial image fed to the model and the output of the first layer. From latter we can see that the model is looking for a circle with some kind of figure inside. In feature map 4 we could even see a truck in the center of the picture.
![alt text][image13]
![alt text][image14]