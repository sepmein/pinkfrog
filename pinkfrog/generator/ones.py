from pinkfrog.generator import Generator
import tensorflow as tf
import tensorflow_probability as tfp


class Ones(Generator):
    def __init__(self, rate: float = None):
        self.rate = rate
        super(Ones, self).__init__()

    def __call__(self, n, *args, **kwargs):
        if self.rate:
            tensor = tfp.distributions\
                .Bernoulli(probs=self.rate, dtype=tf.float32)\
                .sample(n)
        else:
            tensor = tf.ones(n, dtype=tf.float32)

        return tensor
