"""
Reference:

Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
"""
import mxnet as mx
import os

def get_symbol(num_classes, **kwargs):
    if not os.environ['FRAMEWARK_DIR']:
        raise AssertionError('os environment FRAMEWARK_DIR is needed')
    prob, arg_dict, axu_param = mx.model.load_checkpoint(
                            os.path.join(os.environ['FRAMEWARK_DIR'],
                                        "models/training_model/ResNet101_mxnet"),
                            0)
    return prob
