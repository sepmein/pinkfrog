from pinkfrog.layer import Layer
from typing import Any, List
import tensorflow as tf


class SusceptibleInfectiousProbability(Layer):
    def __init__(
            self,
            trainable=False,
            name="SI_Model_Probability_Calculator",
            dtype=tf.float32,
            dynamic=False,
            **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self,
             tensor: tf.Tensor,
             slice_index: int,
             related_index: List[int],
             *args) -> Any:
        target_tensor = super()._get_target_tensor(
            tensor=tensor,
            slice_index=slice_index
        )

        s = tf.reduce_sum(target_tensor)
        n = tf.size(target_tensor, out_type=tf.dtypes.float32)
        i = n - s
        result = i * (s / n)
        return tensor, slice_index, related_index, result
