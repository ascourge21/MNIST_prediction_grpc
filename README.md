# MNIST_prediction_grpc
A simple grpc based implementation of server/client routine that classifies MNIST images.


server script: classifier_server.py

client script: classifier_client.py (an MNIST image pops up after you run this, need to close it first before a request is sent to the server because pyplot.show() blocks the process)


proto file: clssifier.proto


train_mnist_cnn.py was used to train a simple convolutional (conv) neural network with two conv layers and 1 dense layer. 


