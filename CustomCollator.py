import collections
import torch

class CustomCollator(object):
    def __init__(self, *params):
        self.params = params
    def __call__(self, batch):
        self.params.process
        if len(self.params) == 1:
            batch = [batch, ]
        else:
            batch = list(zip(*batch))

        tensors = []

        for it, i in enumerate(batch):
            tensor = self.params.process(i['text'])
        for field, data in zip(self.params.values(), batch):
            tensor = field.process(data)
            if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                tensors.extend(tensor)
            else:
                tensors.append(tensor)

        if len(tensors) > 1:
            return tensors
        else:
            return tensors[0]