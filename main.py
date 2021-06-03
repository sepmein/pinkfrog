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
