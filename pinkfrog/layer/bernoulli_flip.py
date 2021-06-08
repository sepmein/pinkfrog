from typing import Any, Tuple, List
from pinkfrog.layer import Layer
import tensorflow as tf
import tensorflow_probability as tfp


class BernoulliFlip(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self,
             tensor: tf.Tensor,
             slice_index: int,
             related_tensor_index: List[int],
             probability: tf.Tensor = None,
             *args: Tuple,
             **kwargs: Any) -> Tuple[tf.Tensor, Any]:
        target_tensor = super()._get_target_tensor(
            tensor=tensor,
            slice_index=slice_index
        )
        # manipulate
        # transform target tensor to tf.int32
        target_tensor = tf.cast(target_tensor, dtype=tf.int32)
        # generate using bernoulli distribution
        tensor_size = tensor.shape
        tensor_columns = tensor_size[1]
        sample = tfp.distributions.Bernoulli(probs=probability,
                                             dtype=tf.int32) \
            .sample(tensor_columns)
        # cal bitwise xor
        to_update_tensor = tf.bitwise.bitwise_xor(target_tensor, sample)
        to_update_tensor = tf.cast(to_update_tensor, tf.float32)
        # update original tensor
        updated_tensor = super()._update_tensor(
            tensor=tensor,
            slice_index=slice_index,
            to_update_tensor=to_update_tensor
        )
        return updated_tensor, slice_index, related_tensor_index
