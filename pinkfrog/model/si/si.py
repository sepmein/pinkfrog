from pinkfrog.targetgroup import TargetGroup
from pinkfrog.state import State
from pinkfrog.transistor import Transistor
from pinkfrog.layer import Layer
from pinkfrog.layer import Multiply
from pinkfrog.layer import Count
from pinkfrog.generator import Ones
from pinkfrog.layer import BernoulliFlip
import tensorflow_probability as tfp

tfd = tfp.distributions


class SI:
    """
    Susceptible Infectious Model - the simplest infectious disease model
    """

    def __init__(self,
                 n: int = 1000,
                 length: float = 365,
                 possibility_of_infection: float = 0.01
                 ):
        self.length = length
        self.target_group = TargetGroup(
            name='SI model',
            n=n
        )

        transistor = Transistor()

        count_susceptible = Count('Susceptible')
        transistor.add(count_susceptible)

        contact_rate = 4
        contact = Multiply(contact_rate)
        transistor.add(contact)

        infect_rate = 0.5
        infected = Multiply(infect_rate)
        transistor.add(infected)

        infect_rate = Multiply(1.0 / n)
        transistor.add(infect_rate)

        bernoulli_flip = BernoulliFlip()
        transistor.add(bernoulli_flip)

        transistor.add(count_susceptible)

        susceptible_state = State(
            name='Susceptible',
            transistor=transistor,
            generator=Ones()
        )
        self.target_group.add_state(susceptible_state)
        pass
