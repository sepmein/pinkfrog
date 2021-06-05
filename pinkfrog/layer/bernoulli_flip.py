from typing import Any
from pinkfrog.layer import Layer
import tensorflow as tf
import tensorflow_probability as tfp


class BernoulliFlip(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, *args: Any, **kwds: Any) -> Any:
        ## seems it's bug of keras, should use args[0] instead of args here
        tensor, tensor_slice_index, probability = args[0]
        target_tensor = super()._get_slice(tensor, tensor_slice_index)
        target_tensor = tf.cast(target_tensor, dtype=tf.int32)
        # generate sample
        # using bernoulli distribution
        tensor_size = tensor.shape
        tensor_columns = tensor_size[1]
        sample = tfp.distributions.Bernoulli(probs=probability, dtype=tf.int32).sample(
            tensor_columns
        )
        # cal bitwise xor
        to_update_tensor = tf.bitwise.bitwise_xor(target_tensor, sample)
        to_update_tensor = tf.cast(to_update_tensor, tf.float32)
        # update original tensor
        updated_tensor = super()._update_tensor(
            tensor, tensor_slice_index, to_update_tensor
        )
        return updated_tensor, tensor_slice_index
