from pinkfrog.layer.layer import Layer


class Add(Layer):
    def __init__(self, to_add: float, **kwargs):
        self.to_add = to_add
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        # get args
        tensor, tensor_slice_index = args[0]
        # get target tensor
        target_tensor = super()._get_slice(tensor, tensor_slice_index)
        # manipulate
        to_update_tensor = target_tensor + self.to_add
        # update
        updated_tensor = super()._update_tensor(tensor, tensor_slice_index, to_update_tensor)
        return updated_tensor, tensor_slice_index
