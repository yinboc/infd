import torch


models = dict()


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def load_sd_from_ckpt(ckpt, pop_keys=None):
    sd = torch.load(ckpt, map_location='cpu')['model']['sd']
    if pop_keys is not None:
        pop_keys_dot = tuple([_ + '.' for _ in pop_keys])
        pop_keys = set(pop_keys)
        sd_keys = list(sd.keys())
        for k in sd_keys:
            if k in pop_keys or k.startswith(pop_keys_dot):
                sd.pop(k)
    return sd


def make(spec, load_sd=False):
    if load_sd:
        sd = spec['sd']
    elif spec.get('load_ckpt') is not None:
        sd = load_sd_from_ckpt(spec['load_ckpt'], spec.get('load_ckpt_pop_keys'))
        load_sd = True

    if spec.get('args') is None:
        spec['args'] = dict()
    model = models[spec['name']](**spec['args'])
    if load_sd:
        model.load_state_dict(sd, strict=False)
    return model


@register('identity')
def make_identity():
    return torch.nn.Identity()
