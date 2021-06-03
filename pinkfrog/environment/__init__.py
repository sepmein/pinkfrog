from typing import Any


class Environment:
    """Docstring for Environment."""

    def __init__(self):
        """TODO: to be defined.
        :Docstring for Environment.: TODO
        """
        self.time = 0
        self.target_group = []
        Environment.__init__(self)

    def add_target_group(self, target_group: Any) -> None:
        self.target_group.append(target_group)

    def next(self, steps: int):
        for _ in range(steps):
            self.time = self.time + 1
            for i in self.target_group:
                self.target_group[i].next()
