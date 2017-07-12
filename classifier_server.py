"""
    this is the server side code, that loads a trained mnist model and responds
    to client's classification request
"""

import grpc
from concurrent import futures
import time
import classifier_pb2
import classifier_pb2_grpc
import tensorflow as tf
import numpy as np


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
MNIST_IM_SZ = 784  # MNIST vectorized data size
BIT_LEN = 8  # for 8 bit sampling
MAX_INT = 255  # Maximum intensity value with 8 bits


class Classifier(classifier_pb2_grpc.ClassifierServicer):
    def __init__(self):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph('mnist_cnn_model.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint('./'))

        # load objects from the session
        graph = tf.get_default_graph()
        self.y_out = graph.get_tensor_by_name("y_out:0")
        self.x = graph.get_tensor_by_name("input:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")

    def classify_image(self, request, context):
        im_array = self.convert_8bit_string_to_array(request.im_str)
        pred_val = np.argmax(self.sess.run(self.y_out, {self.x: im_array, self.keep_prob: 1.0}))
        print("predicted val: " + str(pred_val))

        return classifier_pb2.ClassifyReply(message='prediction for this image is: ' + str(pred_val))

    def convert_8bit_string_to_array(self, im_8_bit_str):
        im_array = np.zeros((1, MNIST_IM_SZ))
        for i in range(0, int(len(im_8_bit_str) / BIT_LEN)):
            im_array[0][i] = int(im_8_bit_str[i * BIT_LEN:(i + 1) * BIT_LEN], 2) / MAX_INT
        return im_array


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    classifier_pb2_grpc.add_ClassifierServicer_to_server(Classifier(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
