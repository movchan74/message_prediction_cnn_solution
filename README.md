# The solution of "message prediction" contest from unpossib.ly based on fully convolutional network (FCN).

This solution uses two architectures: ResNet and U-Net.

data_preparation_resnet.py - script for prepare data for ResNet.
data_preparation_unet.py - script for prepare data for U-Net.

train_resnet.py - script for train ResNet model.
train_unet.py - script for train U-Net model.

At prediction phase 1 ResNet model and 2 U-Net models (same model but different epoch).

Prediction scripts: 
- resnet_prediction.py
- unet_prediction.py
- unet_prediction_2.py

Checkpoints for prediction can be download here - https://goo.gl/3j0E6K.
