import tensorflow as tf
from typing import Any, List
from pinkfrog.layer import Layer
from numpy import arange


class Transistor:
    """
    Transistor class is a wrapper around a PinkFrag Layer
    """
    transistor: List[Layer]

    def __init__(self):
        self.transistor = []
        pass

    def __call__(self, *args, **kwargs):
        next_layer = None
        for layer in self.transistor:
            if next_layer is None:
                next_layer = layer(*args, **kwargs)
            else:
                next_layer = layer(*next_layer)

        return next_layer

    def add(self, layer: Layer):
        """
        Transistor.add function could add a PinkFrog Layer into its internal chain.
        Layers then will be executed by the TargetGroup.next() function
        """
        self.transistor.append(layer)
        return self

