# TODO: decide which framework to use for Matrix manipulation. Tensorflow or Pytorch
# TO LEARN: How to create a class in Python.
# TODO: create a human class
# TODO: create an attribute class

from typing import Any, Callable, List

import tensorflow as tf
from tensorflow.python.framework import indexed_slices
from tensorflow.python.ops.array_ops import stack
import tensorflow_probability as tfp
import numpy as np


class TargetGroup(object):
    """Docstring for TargetGroup."""

    def __init__(self, name: str, n: int) -> None:
        """TODO: to be defined.

        :Docstring for TargetGroup.: TODO

        """
        self.name = name
        self.n = n
        self.tensor = None
        self.state = {}

    def add_state(self, state: Any) -> None:
        for key, value in self.state.items():
            if state.name == key:
                raise NameError(
                    "pf.add_state, state name:",
                    state.name,
                    " has already in the group.",
                )
        initiated = state.generator(self.n)
        self.state[state.name] = {
            "index": len(self.state),
            "transistor": state.transistor,
        }
        if self.tensor is not None:
            self.tensor = tf.stack([self.tensor, initiated])
        else:
            self.tensor = initiated

    def remove_state(self, state: Any) -> None:
        pass

    def next(self) -> None:
        for key, state in self.state.items():
            tensor_slice_index = state["index"]
            transistor = state["transistor"]
            self.tensor = transistor(self.tensor, tensor_slice_index)

    def _get_index_by_name_(self, name_list: List):
        # get index
        index = []
        for name in name_list:
            if name in self.state:
                index.append(self.state[name].index)
            else:
                raise Exception("Name not in state list")
        # get tensor by index
        return index


class State:
    def __init__(
        self,
        name: str,
        transistor: Callable,
        generator: Callable,
        related_state: List = None,
    ) -> None:
        self.name = name
        self.related_state = related_state
        self.transistor = transistor
        self.generator = generator


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
            # if not a tensor, then transform it to a tensor
            if not tf.is_tensor(tensor_slice_index):
                tensor_slice_index = tf.convert_to_tensor(tensor_slice_index)

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


class Bernoulli_Flipper(Transistor):
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


class Generator:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def ones():
        def generate_ones(n):
            return tf.ones(n, dtype=tf.float32)

        return generate_ones


class Environment:
    """Docstring for Environment."""

    def __init__(self):
        """TODO: to be defined.

        :Docstring for Environment.: TODO

        """
        self.time = 0
        self.target_group = []
        Environment.__init__(self)

    def add_target_group(self, target_group: Any) -> None:
        self.target_group.append(target_group)

    def next(self, steps: int):
        for _ in range(steps):
            self.time = self.time + 1
            for i in self.target_group:
                self.target_group[i].next()


# class Individual():
#    """
#    Individual is the basic build block of the whole dynamic system.
#    Individual could have several **states**, each state is essentially a number, for example: a human has a state called age, and it is a number from 0 to some number.
#    Could a state be represented by a group of numbers? For now, I don't find the case.
#    Here I want to propose an example of dynamic and stochastic birth system to simulate the birth and death of population
#    """
#    def __init__(self) -> None:
#        self.states = []
#        pass
#
#    def set(self, state:Any) -> Any:
#        self.states.append(state)
#        return self
#
#
# class Group():
#    def __init__(self, individual: Individual, number: int) -> None:
#        if (type(individual) is not Individual):
#            raise Exception()
#        pass
#
# class Disease():
#  def __init__(self, name):
#        # disease name
#        self.name = name
#
# class Disease_COVID(Disease):
#    def __init__(self):
#        # inherit from disease
#        super(Disease, self).__init__()
#
# class People():
#    def __init__(self):
#        return
#
