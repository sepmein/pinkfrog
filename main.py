# TODO: decide which framework to use for Matrix manipulation. Tensorflow or Pytorch
# TO LEARN: How to create a class in Python.
# TODO: create a human class
# TODO: create an attribute class

from typing import Any, Callable

import tensorflow as tf
import tensorflow_probability as tfp

class Individual():
    """
    Individual is the basic build block of the whole dynamic system.
    Individual could have several **states**, each state is essentially a number, for example: a human has a state called age, and it is a number from 0 to some number.
    Could a state be represented by a group of numbers? For now, I don't find the case.
    Here I want to propose an example of dynamic and stochastic birth system to simulate the birth and death of population
    """
    def __init__(self) -> None:
        self.states = []
        pass

    def set(self, state:Any) -> Any:
        self.states.append(state)
        return self

class State():
    def __init__(self, name: str, value: Any, next: Callable ) -> None:
        self.name = name
        self.value = value
        self.next = next
        return self

class Group():
    def __init__(self, individual: Individual, number: int) -> None:
        if (type(individual) is not Individual):
            raise Exception()
        pass

class Disease():
  def __init__(self, name):
        # disease name
        self.name = name 

class Disease_COVID(Disease):
    def __init__(self):
        # inherit from disease
        super(Disease, self).__init__()

class People():
    def __init__(self):
        return

Spencer = Individual()
def age_next(age):
    return age + 1

age = State('age', 33, age_next)
