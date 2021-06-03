import tensorflow as tf


class Generator:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def ones():
        def generate_ones(n):
            return tf.ones(n, dtype=tf.float32)

        return generate_ones

