# TODO: decide which framework to use for Matrix manipulation. Tensorflow or Pytorch
# TO LEARN: How to create a class in Python.
# TODO: create a human class
# TODO: create an attribute class

import tensorflow as tf

class Disease():
    def __init__(self, name):
        # disease name
        self.name = name 

class Disease_COVID(Disease):
    def __init__(self):
        # inherit from disease
        super(Disease, self).__init__()

class Person():
    def __init__(self):
        return

