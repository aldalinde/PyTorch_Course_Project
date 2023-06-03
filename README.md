# PyTorch_Course_Project
Hand gesture recognition model traind on database, composed by a set of near infrared images acquired by the Leap Motion sensor.


model_Net.py 
contains two classes:
- CustomDatasetFromImages with torch.utils.data.Dataset as parent class for taining images processing 
- Net torch.nn.Module to build a NN model
and variables with class names dictionary, learning rate, loss function

data_to_numpy.ipynb
processed given Leap Motion sensor dataset and stored images as numpy array in a file

training_model.ipynb
downloads numpy images, formes Dataset and loads it to the Net model. Training done in 10 epochs with
accuracy 99.8% on test dataset. Net model weights saved to file/

gesture_model_use.ipynb
containes several functions for getting from file, visualising, processing new images, predicting
their class and showing prediction in output
