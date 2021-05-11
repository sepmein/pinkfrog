import main as pf
import tensorflow_probability as tfp
import pytest

def test_import():
    Uganda = pf.TargetGroup(name='Uganda', n=1000)
    assert (Uganda.name == 'Uganda')
    assert (Uganda.n == 1000)


def test_add_state():
    Uganda = pf.TargetGroup(name='Uganda', n=100)
    age = pf.State(name='age',
                   transistor=pf.Transistor.add(number=1),
                   generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample
                   )
    Uganda.add_state(age)
    Uganda.next()
    assert (len(Uganda.state) == 1)

def test_add_two_state():
    Uganda = pf.TargetGroup(name='Uganda', n=100)
    age = pf.State(name='age',
                   transistor=pf.Transistor.add(number=1),
                   generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample
                   )
    dummy_state = pf.State(name='dummy',
                           transistor=pf.Transistor.add(number=100),
                           generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample
                           )
    Uganda.add_state(age)
    Uganda.add_state(dummy_state)
    assert (len(Uganda.state) == 2)
    assert (Uganda.tensor.shape == [100, 2])
    Uganda.next()

def test_add_two_same_name_state():
    Uganda = pf.TargetGroup(name='Uganda', n=100)
    age = pf.State(name='age',
                   transistor=pf.Transistor.add(number=1),
                   generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample
                   )
    dummy_state = pf.State(name='age',
                           transistor=pf.Transistor.add(number=100),
                           generator=tfp.distributions.NegativeBinomial(total_count=100, probs=0.4).sample
                           )
    with pytest.raises(NameError):
        Uganda.add_state(age)
        Uganda.add_state(dummy_state)