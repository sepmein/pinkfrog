from pinkfrog.layer.layer import Layer
import tensorflow as tf


class Multiply(Layer):
    def __init__(self, to_multiply: float, **kwargs):
        self.to_multiply = to_multiply
        super().__init__(**kwargs)

    def call(self,
             tensor: tf.Tensor,
             slice_index: int,
             related_index: int,
             result: tf.Tensor = None,
             *args, **kwargs):
        # get args
        # expecting the first element be a tensor and the second element be the slice index of targeting tensor
        target_tensor = super()._get_target_tensor(
            tensor=tensor,
            slice_index=slice_index
        )
        if result:
            # the first three is always tensor, slice_index and related_index
            # So in this situation, the forth will be the result from last layer
            # We will be adding last layers result with our number here
            result = tf.cast(result, tf.float32)
            result = result * self.to_multiply
            return tensor, slice_index, related_index, result
        elif len(args) == 3:
            # no result returned from the last layer
            # then we multiply the target tensor
            to_update_tensor = target_tensor * self.to_multiply
            # update
            updated_tensor = super()._update_tensor(
                tensor=tensor,
                slice_index=slice_index,
                to_update_tensor=to_update_tensor
            )
            return updated_tensor, slice_index, related_index
