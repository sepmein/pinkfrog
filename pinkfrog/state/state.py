from typing import List
from pinkfrog.transistor import Transistor
from pinkfrog.generator import Generator


class State:
    def __init__(
            self,
            name: str,
            transistor: Transistor,
            generator: Generator,
            related_state: List = None,
    ) -> None:
        if related_state is None:
            related_state = []
        self.name = name
        self.transistor = transistor
        self.generator = generator
        self.related_state = related_state
