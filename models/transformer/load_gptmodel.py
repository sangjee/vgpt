'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import logging
import torch

logger = logging.getLogger(__name__)
index_path = './data/index_gpt2withhallym.txt'

def load_weight(model, state_dict):

    # -----
    with open(index_path, 'r') as f:
        data = f.readlines()
    index_list = list(map(int,data))
    
    custom_weight = torch.zeros(model.transformer.wte.weight.shape)
    for i in range(len(index_list)):
        custom_weight[i,:] = state_dict['wte.weight'][index_list[i],:]

    old_keys = []
    new_keys = []
    for key in state_dict.keys():


        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []



    # remove embedding and positioning layer
    for param_tensor in state_dict.copy():
        if 'wte.weight' in param_tensor :
            del(state_dict[param_tensor])

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix="", custom_weight=None):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        model.transformer.wte.weight = torch.nn.Parameter(custom_weight)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_model = model
    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    load(start_model, prefix="", custom_weight=custom_weight)

    # Make sure we are still sharing the output and input embeddings after loading weights
    model.set_tied()
    return model