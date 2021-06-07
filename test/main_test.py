import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import pinkfrog as pf
import pinkfrog.layer

import pinkfrog.transistor as pft
from pinkfrog.targetgroup import TargetGroup
from pinkfrog.transistor import Transistor
from pinkfrog.state import State
from pinkfrog.layer import Add
from pinkfrog.generator import Ones
from pinkfrog.layer import BernoulliFlip


def test_import():
    Uganda = TargetGroup(name="Uganda", n=1000)
    assert Uganda.name == "Uganda"
    assert Uganda.n == 1000


def test_add_state():
    Uganda = TargetGroup(name="Uganda", n=100)
    transistor = Transistor()
    transistor.add(Add(1))
    age = State(
        name="age",
        transistor=transistor,
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    Uganda.add_state(age)
    Uganda.next()
    assert len(Uganda.state) == 1


def test_add_two_state():
    Uganda = TargetGroup(name="Uganda", n=100)
    transistor = Transistor()
    transistor.add(Add(1))
    age = State(
        name="age",
        transistor=transistor,
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    transistor_dummy = Transistor()
    transistor.add(Add(100))
    dummy_state = State(
        name="dummy",
        transistor=transistor_dummy,
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    Uganda.add_state(age)
    Uganda.add_state(dummy_state)
    assert len(Uganda.state) == 2
    assert Uganda.tensor.shape == [2, 100]
    Uganda.next()


def test_add_two_same_name_state():
    Uganda = TargetGroup(name="Uganda", n=100)
    age_transistor = Transistor()
    age_transistor.add(Add(1))
    age = State(
        name="age",
        transistor=age_transistor,
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    dummy_transistor = Transistor()
    dummy_transistor.add(Add(100))
    dummy_state = State(
        name="age",
        transistor=dummy_transistor,
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    with pytest.raises(NameError):
        Uganda.add_state(age)
        Uganda.add_state(dummy_state)


#
# def test_sip():
#     sip = pf.Susceptible_Infectious_Probability()
#     to_test = tf.constant([1., 0.])
#     *arg, result = sip(to_test, 0)
#     assert (result.numpy() == 0.5)
#
#
# def test_bernoulli_flip_with_sip():
#     to_test = tfp.distributions.Bernoulli(probs=0.1, dtype=tf.float32).sample([2, 100])
#     sip = pf.Susceptible_Infectious_Probability()(to_test, 0)
#     bernoulli_flip = pf.Bernoulli_Flipper()(sip)
#     tensor, index = bernoulli_flip
#
#
# def test_init_transistor_without_call():
#     input = tf.keras.Input(shape=(32,))
#     sip = pf.Susceptible_Infectious_Probability()(input, 1)
#     bernoulli_flip = pf.Bernoulli_Flipper()(sip)
#     print(bernoulli_flip)


def test_si_model():
    SI = TargetGroup(name="si", n=10000)
    transistor = Transistor()
    transistor.add(pf.layer.Add(1))
    age = State(
        name="age",
        transistor=transistor,
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )

    # transitor
    # input(what layer should know about the tensor): 'disease_susceptible'
    # fn: calculate_si_percentage, bernoulli_flip
    disease_state_transistor = Transistor()
    disease_state_transistor\
        .add(pf.layer.Count())\
        .add(pf.layer.Multiply(4 / 100 * 0.5))\
        .add(BernoulliFlip())
    disease_suspectible = State(
        name="disease_susceptible",
        transistor=disease_state_transistor,
        generator=pf.generator.Ones(rate=0.01)
    )

    # add state
    SI.add_state(age)
    SI.add_state(disease_suspectible)

    # add another state
    assert len(SI.state) == 2
    print(SI.tensor)
    SI.next()
    print(SI.tensor)
    SI.next()
    print(SI.tensor)
    SI.next()
    print(SI.tensor)
    SI.next()
    print(SI.tensor)


def test_add_multiple_correlated_state():
    dummy = TargetGroup(name="dummy", n=100)

    def transistor_fn():
        def return_fn(tensor, self_state_index, related_tensor_index_list):
            pass

        pass

    pass
    # a = State(name='a',
    #              related_state=['b'],
    #              generator=pf.Generator.ones,
    #              layer=)

    # b = State(name='b',
    #              related_state=['a'],
    #              generator=pf.Generator.ones,
    #              layer= )
