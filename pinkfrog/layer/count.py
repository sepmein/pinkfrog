from typing import List
from pinkfrog.layer import Layer
import tensorflow as tf
from tensorflow import math


class Count(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self,
             tensor: tf.Tensor,
             slice_index: int,
             related_index: List[int], *args, **kwargs):
        # get target tensor
        target_tensor = super()._get_target_tensor(
            tensor=tensor,
            slice_index=slice_index
        )
        # manipulate
        result = math.count_nonzero(target_tensor)
        return tensor, slice_index, related_index, result
