from pinkfrog.layer import Layer
from tensorflow import math


class Count(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        # get args
        tensor, tensor_slice_index, related_index = args
        # get target tensor
        target_tensor = super()._get_slice(tensor, tensor_slice_index)
        # manipulate
        result = math.count_nonzero(target_tensor)
        return tensor, tensor_slice_index, related_index, result
