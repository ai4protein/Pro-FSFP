import torch
from peft.tuners.lora import LoraLayer

lora_modules = ['lora_A',
                'lora_B',
                'lora_embedding_A',
                'lora_embedding_B',
                'lora_dropout']

def pack_lora_layers(model):
    '''
    Find Lora modules in a PeftModel and return them as a single ModuleList.
    Return their corresponding names (same as the ones from model.named_modules()) as well.
    '''
    packed_layers = torch.nn.ModuleList([])
    names = []
    key_list = [key for key, _ in model.named_modules() if 'lora' not in key]
    for key in key_list:
        target = model.get_submodule(key)
        if isinstance(target, LoraLayer):
            for attr in lora_modules:
                if model.active_adapter in getattr(target, attr):
                    names.append('.'.join([key, attr]))
                    packed_layers.append(getattr(target, attr))
    return names, packed_layers

def replace_modules(model, module_names, new_modules):
    '''
    Replace specific modules in the model.
    Names of the new modules should be given by model.named_modules()
    '''
    for name, module in zip(module_names, new_modules):
        name = name.split('.')
        parent = model.get_submodule('.'.join(name[:-1]))
        target_name = name[-1]
        setattr(parent, target_name, module)
