from typing import Callable, List


class State:
    def __init__(
            self,
            name: str,
            transistor: Callable,
            generator: Callable,
            related_state: List = None,
    ) -> None:
        self.name = name
        self.related_state = related_state
        self.transistor = transistor
        self.generator = generator
