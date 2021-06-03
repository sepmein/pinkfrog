from typing import Any, Callable
import tensorflow as tf
import numpy as np


class Transistor(tf.keras.layers.Layer):
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
    def input(*names):
        return names

    @staticmethod
    def layer(fn) -> Callable:
        def layer_fn(input):
            return fn(input)

        return layer_fn

    @staticmethod
    def _get_slice(*args):
        tensor, tensor_slice_index = args
        dimension = len(tensor.shape)
        print(dimension)
        if dimension == 1:
            return tensor
        elif dimension == 2:
            return tensor[tensor_slice_index, :]

    @staticmethod
    def _update_tensor(tensor, tensor_slice_index, to_update_tensor):
        dimension = tf.size(tensor.shape)
        if dimension == 1:
            # 1d updates
            indices = np.arange(tensor.shape[0])
            indices = indices.reshape([-1, 1])
            return tf.tensor_scatter_nd_update(tensor, indices, to_update_tensor)
        elif dimension == 2:
            # row update
            indices = tf.constant([[tensor_slice_index.numpy()]])
            to_update_tensor = tf.reshape(to_update_tensor, [1, -1])
            return tf.tensor_scatter_nd_update(tensor, indices, to_update_tensor)

    @staticmethod
    def add(number: int) -> Callable:
        def returned_add(tensor, tensor_slice_index):
            target_tensor = Transistor._get_slice(tensor, tensor_slice_index)
            to_update_tensor = target_tensor + number
            return Transistor._update_tensor(
                tensor, tensor_slice_index, to_update_tensor
            )

        #            if dimension == 1:
        #                return tensor + number
        #            elif dimension == 2:
        #                data_width = tensor.shape[1]
        #                to_add_tensor = np.zeros(data_width)
        #                to_add_tensor[tensor_slice_index] = number
        #                result_tensor = tf.math.add(tensor, to_add_tensor)
        #                return result_tensor
        #
        return returned_add

    @staticmethod
    def multiply(number: int) -> Callable:
        def returned_multiply(tensor, tensor_slice_index):
            target_tensor = Transistor._get_slice(tensor, tensor_slice_index)
            to_update_tensor = target_tensor * number
            return Transistor._update_tensor(
                tensor, tensor_slice_index, to_update_tensor
            )

        return returned_multiply

    @staticmethod
    def bernoulli_flip(logits=None, probs=None) -> Callable:
        def return_bernoulli_flip(tensor, tensor_slice_index):
            # get target tensor, cast into int32
            target_tensor = Transistor._get_slice(tensor, tensor_slice_index)
            target_tensor = tf.cast(target_tensor, dtype=tf.int32)
            # generate sample
            # using bernoulli distribution
            tensor_size = tensor.shape
            tensor_columns = tensor_size[1]
            sample = tfp.distributions.Bernoulli(
                logits=logits, probs=probs, dtype=tf.int32
            ).sample(tensor_columns)
            # cal bitwise xor
            to_update_tensor = tf.bitwise.bitwise_xor(target_tensor, sample)
            to_update_tensor = tf.cast(to_update_tensor, tf.float32)
            # update original tensor
            return Transistor._update_tensor(
                tensor, tensor_slice_index, to_update_tensor
            )

        return return_bernoulli_flip


class Layer(Transistor):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # get target tensor by slice index
        target_tensor = super()._get_slice(args)
        return target_tensor



class Susceptible_Infectious_Probability(Transistor):
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
        S = tf.reduce_sum(target_tensor)
        N = tf.size(target_tensor, out_type=tf.dtypes.float32)
        I = N - S
        result = I * (S / N)
        return *args, result
