from typing import Any, Callable
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

    def call(self, *args: Any, **kwds: Any) -> Any:
        pass

    @staticmethod
    def _get_slice(*args):
        tensor, tensor_slice_index = args
        if not tf.is_tensor(tensor):
            raise Exception("Layer._get_slice, target tensor type should be a tensor, but is: ", type(tensor))
        dimension = len(tensor.shape)
        if dimension == 1:
            return tensor
        elif dimension == 2:
            return tensor[tensor_slice_index, :]

    @staticmethod
    def _update_tensor(tensor, tensor_slice_index, to_update_tensor):
        global indices
        dimension = tf.size(tensor.shape)
        if dimension == 1:
            # 1d updates
            indices = np.arange(tensor.shape[0])
            indices = indices.reshape([-1, 1])
            return tf.tensor_scatter_nd_update(tensor, indices, to_update_tensor)
        elif dimension == 2:
            # row update
            # detect type of the index
            if type(tensor_slice_index) is int:
                indices = tf.constant([[tensor_slice_index]])
            elif type(tensor_slice_index) is tf.constant:
                indices = tf.constant([[tensor_slice_index.numpy()]])
            to_update_tensor = tf.reshape(to_update_tensor, [1, -1])
            return tf.tensor_scatter_nd_update(tensor, indices, to_update_tensor)

    def _get_target_tensor(self, *args):
        tensor, tensor_slice_index, related_index = self._unpack_args(*args)
        target_tensor = self._get_slice(tensor, tensor_slice_index)
        return target_tensor

    def _get_related_tensor(self, *args):
        tensor, tensor_slice_index, related_index = self._unpack_args(*args)
        related_tensor = []
        for index in related_index:
            related_tensor.append(self._get_target_tensor(tensor, index, ))


    @staticmethod
    def _unpack_args(*args):
        tensor, tensor_slice_index, related_index = args
        return tensor, tensor_slice_index, related_index

    @staticmethod
    def input(*names):
        return names


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

    def call(self, *args) -> Any:
        target_tensor = super()._get_slice(*args)
        s = tf.reduce_sum(target_tensor)
        n = tf.size(target_tensor, out_type=tf.dtypes.float32)
        i = n - s
        result = i * (s / n)
        return *args, result
