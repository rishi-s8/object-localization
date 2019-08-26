# Object Localization.
## Flipkart GRiD Machine Learning Challenge
## TEAM: YOLO
### GPU: NVIDIA GTX 1080 - 8GB


## Steps involved:
1. Train an encoder-decoder(autoencoder) network on the given dataset.
    - Source file ```flip.py```
    - Normalize pixel values between 0-1.
    - Normalize ground truth between 0-1.
    - Encoder network comprising of 4 blocks of multiple Convolution layers, Pooling and Batch Normalization
    - Decoder network comprises of 4 blocks of combination of transposed convolution layers, conv layers and batch normalization
    - Use loss function: mean_squared_error
    - Use optimizer: 'adadelta'
    - Train overnight
    - Save the model weights

2. Load the model weights and make the encoder non-trainable. Remove the decoder and attach a Regression network after Encoder
    - Source file ```regressor.py```
    - Since the auto encoder was trained, the encoder already has the weights to create a feature map containing the necessary features
    - Regression network comprises of conv layers, pooling, batch normalization and at the end, Fully Connected layers
    - Use loss function: mean_squared_error
    - Use optimizer: 'adadelta'
    - Output 4 values, x1,x2,y1,y2 with activation of sigmoid in the last layer
    - Train the regression network for 4 hours.
    - Save model weights

3. Load the model weights and train the model fully(including the encoder network) to tune the complete network to predict 4 values.
    - Source file ```1regressorNew.py``` and ```regressorNew.py```
    - Use loss function: mean_squared_error
    - Use optimizer: Adam(lr=0.0002)
    - Train for 2 hours
    - Source file ```regressorNew.py```
    - Use optimizer: Adam(lr=0.00005)
    - Train for 2 hours

4. The model is ready to predict values.
    - Source file ```RegressorTest.py```
    - In the test script, load images one-by-one.
    - Normalize pixel values between 0-1
    - Predict the values using the model
    - Scale the values up by using the resolution of the image
    - Write to ```test.csv``` file

### Test IOU = 83.46% and IIT Mandi Rank 1