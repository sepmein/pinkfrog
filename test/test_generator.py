import tensorflow as tf
from pinkfrog.generator import Generator
from pinkfrog.generator import Ones


def test_generator_type():
    """
    test if the importation works well
    """
    assert isinstance(Generator(), Generator)


def test_generator_ones():
    """
    test generator ones
    """
    # create ones generator
    n = 10
    ones_generator = Ones()
    generated = ones_generator(n)
    compared = generated.numpy() == 1.0
    assert compared.all()
    # create ones generator with rate parameter
    rate = 0.01
    ones_generator_with_rate = Ones(rate=rate)
    assert ones_generator_with_rate.rate == rate
    # test ones generator with sample
    sampled_tensor = ones_generator_with_rate(n)
    assert sampled_tensor.shape == n
