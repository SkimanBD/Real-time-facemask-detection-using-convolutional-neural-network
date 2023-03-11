# Real-time-facemask-detection-using-convolutional-neural-network

According to the World Health Organization's (WHO) Situation Report, the official Situation Report 205 of the World Health Organization (WHO) states that the coronavirus illness 2019 (COVID-19) has killed over 0.7 million people worldwide and infected over 20 million people. The symptoms are seen by those who have COVID-19 range widely, from mild indications to major sickness. One of them is a respiratory issue like shortness of breath or breathing difficulty. Elderly adults with lung disease may experience major COVID-19 sickness consequences since they seem to be more susceptible. Human coronaviruses 229E, HKU1, OC43, and NL63 are some of the frequent ones that infect people worldwide. Viruses such as 2019-nCoV, SARS-CoV, and MERS-CoV initially infect animals before causing harm to humans. Anyone who comes into contact with someone who has respiratory issues could be exposed to infectious beads. A contaminated person's surroundings can lead to contact transmission since virus- carrying droplets may also land on nearby surfaces.

A clinical mask is important to treat some respiratory viral diseases, such as COVID-19. The general population should be informed about whether to wear a mask for source control or COVID-19 avoidance. Potential benefits of using masks include lowering vulnerability to danger from infectious individuals during the "pre-symptomatic" stage and stigmatizing specific individuals who use masks to stop the spread of viruses. WHO emphasizes that medical masks and respirators for healthcare assistants should be prioritized. Therefore, face mask detection has become a crucial task in the present global society.
Face mask detection entails locating the face and then assessing whether or not it is covered by a mask. The problem is closely related to broad object detection to identify object classes. Face identification is the process of identifying a certain class of entities, namely faces. It has several uses, including autonomous driving, education, spying, and other things. This paper proposes a condensed method to achieve the aforementioned goal using the fundamental Machine Learning (ML) packages TensorFlow, Keras, OpenCV, and Scikit- Learn.

The related research on face mask identification is discussed further below. The nature of the employed dataset is covered in Section III. The packages that were used to construct the suggested model are described in detail in Section IV. A summary of our approach is provided in Section
V. Section VI reports the experimental findings and analysis. Section VII comes to a close and points the way forward for new projects.


Algorithm 1: Face Mask Detection

Input: Dataset including faces with and without masks
Output: categorized image depicting the presence of face mask
1.	for each image in the dataset do
2.	       Visualize the image in two categories and label them
3.	       Convert the RGB image to Gray-scale image
4.	       Resize the gray-scale image into 100 x 100
5.	       Normalize the image and convert it into 4 dimensional array
6.	end
7.	for building the CNN model do
8.	       Add a Convolution layer of 200 filters
9.	       Add the second Convolution layer of 100 filters
10.	       Insert a flatten layer to the network classifier
11.	       Add a Dense layer of 64 neurons
12.	       Add the final Dense layer with 2 outputs for 2 categories
13.	end
14.	Split the data and train the model

2. Development of the software

The software is developed using python programming language. The software will use TensorFlow and keras libraries along with open CV for application. The development phase consists of two main stages. The training and implementing the trained model to application stage. The following subsection discusses the development process.




Training of Model

2.1 Data preprocessing:

The dataset was organized through google images and Kaggle. Two datasets were built labeled with mask and without mask. There are 1900 masked faces and 1900 unmasked faces. A training image generator was used for data augmentation to create more images with alteration making the dataset larger with around 3800 images. This makes the training model robust and increase accuracy. The dataset is divided 80% for training and 20% for testing. Here with the use of mobilenetV2 the base model was loaded using the pertained imagenet function of mobilenetv2.

2.2 Training the model using CNN:

CNN is a Convolutional Neural Network or CNN is a type of artificial neural network, which is widely used for image/object recognition and classification. CNN takes input as a two-dimensional array and works directly on the images rather than concentrating on feature extraction which other neural networks emphasize. MobilenetV2 is a CNN model consisting of 53 layers and 19 blocks is used.

The architecture consists of convolutional layers (conv2D), pooling layers, activation functions and fully connected layers. The trained model uses average pooling for its pooling layers. Activation functions determines which nodes get activated on each layer, the head model is constructed by placing it on top of the base model. We used AveragePooling2D, Flatten, Dense and Dropout. For the hidden layers the ReLU activation function is used for hidden layers whereas for output layer Softmax is used.     TensorFlow and keras are the main building blocks of the model. for compilation we use “Adam” optimizer and “binary cross entropy” as our loss function. A loss function is a function that compares the target and predicted output values; measures how well the neural network models the training data. When training, our goal is to minimize this loss between the predicted and target outputs. An optimizer helps us to reduce the losses. EPOCH was set to 100 which provided the optimum results.

Implementing the trained model
2.3 Face detection using OpenCV
Our aim in this section is to use the webcam of our device and give result according to the trained model. Here the mechanism is to create single snaps of images from the video and compare it with the trained model to give the result during the live stream. With the help of OpenCV we extract the face region of interest. Convert it from BGR to RGB channel ordering, resizing it to 224x224, and preprocess it. Caffe is a deep learning framework that is used to for face detection. It is a powerful object detection technique and can be accessed through OpenCV repository. res10_300x300_ssd_iter_140000.caffemodel provides the weight of the layers and deploy.prototxt file illustrates the architecture. A facemask detector model is created which will use the trained model and give predictions on whether a person is wearing a facemask or not.

2.4 Face mask detection model
The trained model is now implemented in order to determine the presence of facemask on a person’s face. The detector model will run an infinite loop which will enable a video stream to be broken down to millions of frames in order to capture each frame and predict on that particular frame. Due to the model consisting of a face detection model it will be able to detects multiple faces. The model will now be able to detect facemask on person’s face in real time video stream. A snapshot can be taken as a record for the number of people within the stream frame wearing masks.

3. Impact
The developed software is successfully able to detect the presence of mask on a person’s face. This will create a positive impact on containing the spread of covid-19 virus. The facemask detector has been successful in implementing convolutional neural network and face detection technique to achieve its purpose with almost 100% accuracy. In this system, the MobileNetV2 classifier is used with Adam optimizer. The system's performance is a result of its ability to detect face masks in images with several faces from various perspectives. Using MobileNetV2 makes the system efficient and ideal to use install in embedded systems. Our face mask detection is based on the CNN (Convolutional neural network) model, and to determine if a person is wearing a mask or not, we use OpenCV, TensorFlow, Keras, and Python. The facemask detector will allow to analyze public places and monitor the crowd. In order to prevent the spread of virus and to check if someone is wearing a mask or not the detector can be installed in public places like shopping malls, hospitals, schools and stations. The software operator can monitor the crowd remotely where wearing a facemask is mandatory. The facemask detector can play an impactful role in restricting the spread of virus and diseases.


4. Future development
Facemask detector can be used in places where it is essential to wear facemasks. It can be public areas, workshops, factories and even laboratories. The software can be used to count the number of people wearing mask and can be used to statistically analyze the population mean wearing facemask. The software can be installed into CCTV cameras in public areas and can also be used as a security feature in places where facemask is mandatory. An image can be obtained by the operators to check for facemasks. As a security feature it can be connected to any entrance gate where only the people wearing facemask will be granted access. The software can be further tested with different neural network architecture and the dataset can be increase in size for greater efficiency.


References
1.	W.H.O., “Coronavirus disease 2019 (covid-19): situation report,
a.	205” 2020.
2.	B. QIN and D. Li, identifying facemask-wearing condition using image super-resolution with classification network to prevent COVID-19, May 2020, doi:10.21203/rs.3.rs-28668/v1.
3.	“Face Mask Detection”, Kaggle.com, 2020. [Online]. Available:
a.	https://www.kaggle.com/andrewmvd/face-mask-detection. 2020.
4.	Opencv-python-tutroals.readthedocs.io. 2020. Changing Color spaces — OpenCV-Python.
5.	Brownlee, J. (2020, October 18). Softmax Activation Function with Python. Machine Learning Mastery.
https://machinelearningmastery.com/softmax-activation-function- with-python/ 

      6.   Image Classification Using Convolutional Neural Networks Deepika Jaiswal, Sowmya.V, K.P.Soman
