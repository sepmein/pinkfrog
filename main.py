# TODO: decide which framework to use for Matrix manipulation. Tensorflow or Pytorch
# TO LEARN: How to create a class in Python.
# TODO: create a human class
# TODO: create an attribute class

from typing import Any, Callable

import tensorflow as tf
from tensorflow.python.ops.array_ops import stack
import tensorflow_probability as tfp
import numpy as np


class TargetGroup(object):
    """Docstring for TargetGroup. """

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
                raise NameError('pf.add_state, state name:', state.name, ' has already in the group.')
        initiated = state.generator(self.n)
        self.state[state.name] = {
            "name": state.name,
            "index": len(self.state),
            "transistor": state.transistor
        }
        if self.tensor is not None:
            self.tensor = tf.stack([self.tensor, initiated], axis=1)
        else:
            self.tensor = initiated

    def remove_state(self, state: Any) -> None:
        pass

    def next(self) -> None:
        for key, state in self.state.items():
            tensor_slice_index = state['index']
            transistor = state['transistor']
            self.tensor = transistor(self.tensor, tensor_slice_index)


class State():
    def __init__(self,
                 name: str,
                 transistor: Callable,
                 generator: Callable) -> None:
        self.name = name
        self.transistor = transistor
        self.generator = generator


class Transistor():
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add(number: int) -> Callable:

        def returned_add(tensor, tensor_slice_index):
            dimension = len(tensor.shape)
            if dimension == 1:
                return tensor + number
            elif dimension == 2:
                data_width = tensor.shape[1]
                to_add_tensor = np.zeros(data_width)
                to_add_tensor[tensor_slice_index] = number
                result_tensor = tf.math.add(tensor, to_add_tensor)
                return result_tensor

        return returned_add


class Generator():
    def __init__(self) -> None:
        super().__init__()


class Environment():
    """Docstring for Environment. """

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
