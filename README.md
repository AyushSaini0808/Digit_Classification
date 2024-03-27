# Digit_Classification
Image Classification with MNIST Dataset

The MNIST handwritten digit classification problem is a standard dataset used in computer vision and deep learning.
Although the dataset is effectively solved, it can be used as the basis for learning and practicing how to develop, evaluate, and use convolutional deep learning neural networks for image classification from scratch. This includes how to develop a robust test harness for estimating the performance of the model, how to explore improvements to the model, and how to save the model and later load it to make predictions on new data.

MNIST Handwritten Digit Classification Dataset The MNIST dataset is an acronym that stands for the Modified National Institute of Standards and Technology dataset.
It is a dataset of 60,000 small square 28×28 pixel grayscale images of handwritten single digits between 0 and 9.
The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.
It is a widely used and deeply understood dataset and, for the most part, is “solved.” Top-performing models are deep learning convolutional neural networks that achieve a classification accuracy of above 99%, with an error rate between 0.4 %and 0.2% on the hold out test dataset.

# Explanation of code

Step-1 : Preprocessing the images - Converting the data provided in training dataset to float 32 type (standard type used in neural networks).Then we divide each element of X_train with 255 since the dataset stores pixel values between 0 and 255 

Step-2 : Expanding dimensions - since certain NN like CNN require specific number of inputs in the dataset . For example a CNN might excpect to have the following format (height, width and channel) . if the data does not have the three inputs it would display error .Hence, we expanded the dimensions of the training and testing data to assure compatibility with neural networks specification. 

Step-3 : Preprocessing the output - the labels provided in the output variable are easily converted to one-hot encoded format.

Step-4 : Creating the model for classification - The CNN model utilises 5 different types of layers , namely "Conv2D","Flatten","Dropout","Dense","MaxPool2D". 
        
        1) Conv2D - This is used for adding the convolutional layer in the model 
        
        2) MaxPool2D - This defines a max pooling layer. Max pooling reduces the dimensionality of the data by taking the maximum value from a specific window (2x2            in this case) and using that value to represent the entire window. This helps reduce computational cost and introduces some level of translation                    invariance (meaning the model is less sensitive to small shifts in the input).
        
        3) Flatten - This layer flattens the multi-dimensional output of the convolutional layers into a one-dimensional vector. This is necessary because fully               connected layers (next layers) typically require a flattened input.
        
        4) Dropout - This layer introduces Dropout, a regularization technique used to prevent overfitting. Dropout randomly sets a certain percentage (50% in this            case) of neurons to zero during training, forcing the network to learn more robust features that are not dependent on any specific neuron.
Step-5 : Compiling the model -  The model is combined with the help of Adam optimiser and utilies categorical crossentropy to determine loss.

Step-6 : Callbacks - Used for validating the behaviour of our model. The "EarlyStopping" helps prevent overfitting . It monitors a specific metric (often validation loss) during training. If the monitored metric doesn't improve for a certain number of epochs (iterations over the training data), called patience, the callback stops the training process. This helps avoid overfitting by preventing the model from continuing to train on patterns that might not generalize well to new data. 

The "ModelCheckpoint" callback allows you to save the best performing model state (weights and configuration) during training.It allows to define define parameters like the file path to save the model and the metric to monitor. ModelCheckpoint saves the model's weights (or the entire model) whenever the monitored metric (e.g., validation accuracy) improves. By default, it only saves the model with the best performance based on the chosen metric.

Step-7 : Saving the model and evaluating its performance in testing data .

        
