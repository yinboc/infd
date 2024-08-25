datasets = dict()


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(spec):
    if spec.get('args') is None:
        spec['args'] = dict()
    dataset = datasets[spec['name']](**spec['args'])
    return dataset
