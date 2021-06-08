from typing import Tuple, List
from pinkfrog.layer.layer import Layer

import tensorflow as tf


class Add(Layer):
    def __init__(self, to_add: float, **kwargs):
        self.to_add = to_add
        super().__init__(**kwargs)

    def call(self,
             tensor: tf.Tensor,
             slice_index: int,
             related_index: List[int],
             *args, **kwargs) -> Tuple[tf.Tensor, int, List[int]]:
        # get target tensor
        target_tensor = super()._get_target_tensor(
            tensor=tensor,
            slice_index=slice_index
        )
        # manipulate
        to_update_tensor = target_tensor + self.to_add
        # update
        updated_tensor = super()._update_tensor(
            tensor=tensor,
            slice_index=slice_index,
            to_update_tensor=to_update_tensor
        )
        return updated_tensor, slice_index, related_index
