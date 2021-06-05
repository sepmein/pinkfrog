import tensorflow as tf
from typing import Any
from pinkfrog.layer import Layer
from numpy import arange


class Transistor:
    """
    Transistor class is a wrapper around a PinkFrag Layer
    """

    def __init__(self):
        self.fn = []
        pass

    def __call__(self, tensor, indices, *args, **kwargs):
        output = None
        for fn in self.fn:
            output = fn(tensor, indices, output, args, kwargs)
        return output

    def add(self, layer: Layer):
        """
        Transistor.add function could add a PinkFrog Layer into its internal chain.
        Layers then will be executed by the TargetGroup.next() function
        """
        self.fn.append(layer)
        return self

