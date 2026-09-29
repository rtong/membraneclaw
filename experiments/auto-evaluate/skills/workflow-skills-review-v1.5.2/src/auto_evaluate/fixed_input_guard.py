"""Check explicit fixed inputs before a physical call, independently of task family."""
from copy import deepcopy

VERSION = 'fixed-input-guard@0.3.0'


def differences(fixed, arguments, prefix=''):
    """Return missing/changed declared fields; unrelated arguments are allowed.

    Values are already in canonical tool units. No defaults, aliases, tolerance,
    task IDs, or physics assumptions are inferred here.
    """
    if not isinstance(fixed, dict) or not isinstance(arguments, dict):
        raise ValueError('Fixed inputs and tool arguments must be objects')
    out = []
    for key, expected in fixed.items():
        path = f'{prefix}.{key}' if prefix else key
        if key not in arguments:
            out.append(dict(path=path, reason='missing', expected=deepcopy(expected)))
        elif isinstance(expected, dict) and isinstance(arguments[key], dict):
            out.extend(differences(expected, arguments[key], path))
        elif arguments[key] != expected:
            out.append(dict(path=path, reason='changed', expected=deepcopy(expected), actual=deepcopy(arguments[key])))
    return out


def required_paths(values, prefix=''):
    """Required leaf paths, preserving composition structure when values may vary."""
    paths=[]
    for key,value in values.items():
        path=f'{prefix}.{key}' if prefix else key
        paths.extend(required_paths(value,path) if isinstance(value,dict) and value else [path])
    return paths


def has_path(values, path):
    for key in path.split('.'):
        if not isinstance(values,dict) or key not in values:return False
        values=values[key]
    return True


def require_fixed_inputs(fixed, arguments, *, required_fields=()):
    errors = differences(fixed, arguments)
    errors += [dict(path=key, reason="missing_required_argument")
               for key in required_fields if not has_path(arguments,key) and not has_path(fixed,key)]
    if errors:
        raise ValueError('Fixed-input contract violated: '+str(errors))
