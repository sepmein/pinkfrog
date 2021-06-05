from pinkfrog.generator import Generator
import tensorflow as tf


class Ones(Generator):
    def __init__(self):
        super(Ones, self).__init__()

    def __call__(self, *args, **kwargs):
        return tf.ones(n, dtype=tf.float32)
