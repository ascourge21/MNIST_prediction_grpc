"""
    this is the client side code that requests classification on MNIST data.
"""

import grpc
import classifier_pb2
import classifier_pb2_grpc
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt

MNIST_IM_DIM = 28  # MNIST image size
MAX_INT = 255  # Maximum intensity value with 8 bits


# the folder where the data is saved (or gets saved if not already)
mnist_data_path = os.getcwd() + '/MNIST_data'


def view_mnist_image(x_in):
    x_in = np.reshape(x_in, [MNIST_IM_DIM, MNIST_IM_DIM])
    plt.imshow(x_in, cmap='gray')
    plt.show()


def convert_to_8bit_string(im_array):
    im_str_8_bit = ''
    for i in range(len(im_array[0])):
        im_str_8_bit += format(int(im_array[0][i] * MAX_INT), '#010b')[2:]
    return im_str_8_bit


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = classifier_pb2_grpc.ClassifierStub(channel)

    # get data
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

    # keep querying for image until the user wants to step
    while True:
        n_option = int(input('press 0 to stop, any other number to continue: '))
        if n_option == 0:
            break

        # get a random image from the test set and convert to string
        rand_int = np.random.randint(len(mnist.test.images))
        im_num = mnist.test.images[rand_int:rand_int + 1]
        im_str_8_bit = convert_to_8bit_string(im_num)
        view_mnist_image(im_num)

        response = stub.classify_image(classifier_pb2.ClassifyRequest(im_str=im_str_8_bit))
        print("Image received, " + response.message)


if __name__ == '__main__':
    run()
