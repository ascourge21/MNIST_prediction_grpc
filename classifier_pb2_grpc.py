# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import classifier_pb2 as classifier__pb2


class ClassifierStub(object):
  """The greeting service definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.classify_image = channel.unary_unary(
        '/Classifier/classify_image',
        request_serializer=classifier__pb2.ClassifyRequest.SerializeToString,
        response_deserializer=classifier__pb2.ClassifyReply.FromString,
        )


class ClassifierServicer(object):
  """The greeting service definition.
  """

  def classify_image(self, request, context):
    """Sends a greeting
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_ClassifierServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'classify_image': grpc.unary_unary_rpc_method_handler(
          servicer.classify_image,
          request_deserializer=classifier__pb2.ClassifyRequest.FromString,
          response_serializer=classifier__pb2.ClassifyReply.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Classifier', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))