from typing import Any, Callable, List
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


class Layer(tf.keras.layers.Layer):
    def __init__(
            self,
            trainable=False,
            name="Transistor",
            dtype=tf.float32,
            dynamic=False,
            **kwargs
    ):
        super().__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs
        )

    def call(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @staticmethod
    def _get_slice(
            tensor: tf.Tensor,
            slice_index: int,
    ):
        if not tf.is_tensor(tensor):
            raise Exception("Layer._get_slice, target tensor type should be a tensor, but is: ", type(tensor))
        dimension = len(tensor.shape)
        if dimension == 1:
            return tensor
        elif dimension == 2:
            return tensor[slice_index, :]

    @staticmethod
    def _update_tensor(
            tensor: tf.Tensor,
            slice_index: int,
            to_update_tensor: tf.Tensor,
            *args) -> tf.Tensor:
        dimension = tf.size(tensor.shape)
        if dimension == 1:
            # 1d updates
            index = np.arange(tensor.shape[0])
            index = index.reshape([-1, 1])
            return tf.tensor_scatter_nd_update(tensor, index, to_update_tensor)
        elif dimension == 2:
            # row update
            # detect type of the index
            if type(slice_index) is int:
                index = tf.constant([[slice_index]])
            elif type(slice_index) is tf.constant:
                index = tf.constant([[slice_index.numpy()]])
            to_update_tensor = tf.reshape(to_update_tensor, [1, -1])
            return tf.tensor_scatter_nd_update(tensor, index, to_update_tensor)

    def _get_target_tensor(self,
                           tensor: tf.Tensor,
                           slice_index: int
                           ) -> tf.Tensor:
        target_tensor = self._get_slice(tensor, slice_index)
        return target_tensor

    def _get_related_tensor(self,
                            tensor: tf.Tensor,
                            related_index: List[int],
                            *args) -> List[tf.Tensor]:
        # get related tensor using the genetic args
        related_tensor = []
        for index in related_index:
            related_tensor.append(
                self._get_target_tensor(
                    tensor=tensor,
                    slice_index=index
                )
            )
        return related_tensor

    @staticmethod
    def input(*names):
        return names
