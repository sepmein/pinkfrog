from typing import Any, List
import tensorflow as tf
from pinkfrog.state import State


class TargetGroup(object):
    """
    Father class for a target group
    """

    def __init__(self, name: str, n: int) -> None:
        """TODO: to be defined.
        :Docstring for TargetGroup.: TODO
        """
        self.name = name
        self.n = n
        self.tensor = None
        self.state = {}

    def add_state(self, state: State) -> None:
        for key, value in self.state.items():
            if state.name == key:
                raise NameError(
                    "pf.add_state, state name:",
                    state.name,
                    " has already in the group.",
                )
        initiated = state.generator(self.n)
        self.state[state.name] = {
            "index": len(self.state),
            "layer": state.transistor,
            "related": state.related_state
        }
        if self.tensor is not None:
            self.tensor = tf.stack([self.tensor, initiated])
        else:
            self.tensor = initiated

    def remove_state(self, state: Any) -> None:
        pass

    def next(self) -> None:
        for key, state in self.state.items():
            tensor_slice_index = state["index"]
            transistor = state["layer"]
            related_index = self._get_index_by_name_(state["related"])
            self.tensor = transistor(self.tensor, tensor_slice_index, related_index)

    def _get_index_by_name_(self, name_list: List):
        # get index
        index = []
        for name in name_list:
            if name in self.state:
                index.append(self.state[name].index)
            else:
                raise Exception("Name not in state list")
        # get tensor by index
        return index
