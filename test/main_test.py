from tensorflow.python.keras.backend import dtype
import main as pf
import tensorflow as tf
import tensorflow_probability as tfp
import pytest
import numpy as np

pft = pf.Transistor


def test_import():
    Uganda = pf.TargetGroup(name="Uganda", n=1000)
    assert Uganda.name == "Uganda"
    assert Uganda.n == 1000


def test_add_state():
    Uganda = pf.TargetGroup(name="Uganda", n=100)
    age = pf.State(
        name="age",
        transistor=pf.Transistor.add(number=1),
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    Uganda.add_state(age)
    Uganda.next()
    assert len(Uganda.state) == 1


def test_add_two_state():
    Uganda = pf.TargetGroup(name="Uganda", n=100)
    age = pf.State(
        name="age",
        transistor=pf.Transistor.add(number=1),
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    dummy_state = pf.State(
        name="dummy",
        transistor=pf.Transistor.add(number=100),
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    Uganda.add_state(age)
    Uganda.add_state(dummy_state)
    assert len(Uganda.state) == 2
    assert Uganda.tensor.shape == [2, 100]
    Uganda.next()


def test_add_two_same_name_state():
    Uganda = pf.TargetGroup(name="Uganda", n=100)
    age = pf.State(
        name="age",
        transistor=pf.Transistor.add(number=1),
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    dummy_state = pf.State(
        name="age",
        transistor=pf.Transistor.add(number=100),
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )
    with pytest.raises(NameError):
        Uganda.add_state(age)
        Uganda.add_state(dummy_state)


def test_sip():
    sip = pf.Susceptible_Infectious_Probability()
    to_test = tf.constant([1., 0.])
    *arg, result = sip(to_test,0)
    assert (result.numpy() == 0.5)

def test_bernoulli_flip_with_sip():
    to_test = tfp.distributions.Bernoulli(probs=0.1, dtype=tf.float32).sample([2, 100])
    sip = pf.Susceptible_Infectious_Probability()(to_test,0)
    bernoulli_flip = pf.Bernoulli_Flipper()(sip)
    tensor, index = bernoulli_flip

def test_init_transistor_without_call():
    input = tf.keras.Input(shape=(32,))
    sip = pf.Susceptible_Infectious_Probability()(input,1)
    bernoulli_flip = pf.Bernoulli_Flipper()(sip)
    print(bernoulli_flip)

def test_si_model():
    SI = pf.TargetGroup(name="si", n=10)
    age = pf.State(
        name="age",
        transistor=pf.Transistor.add(number=1),
        generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample,
    )


    input = pft.input("disease_susceptible")

    # transitor
    # input(what transistor should know about the tensor): 'disease_susceptible'
    # fn: calculate_si_percentage, bernoulli_flip
    disease_suspectible = pf.State(
        name="disease_susceptible", transistor=transistor, generator=pf.Generator.ones()
    )
    SI.add_state(disease_suspectible)
    comparison = SI.tensor.numpy() == np.ones(10)
    assert comparison.all()

    # add another state
    SI.add_state(age)
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
    dummy = pf.TargetGroup(name="dummy", n=100)

    def transistor_fn():
        def return_fn(tensor, self_state_index, related_tensor_index_list):
            pass

        pass

    pass
    # a = pf.State(name='a',
    #              related_state=['b'],
    #              generator=pf.Generator.ones,
    #              transistor=)

    # b = pf.State(name='b',
    #              related_state=['a'],
    #              generator=pf.Generator.ones,
    #              transistor= )
