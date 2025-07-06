import numpy as np

_original_lstsq = np.linalg.lstsq


# Wrap it
def patched_lstsq(*args, **kwargs):
    if 'rcond' not in kwargs:
        kwargs['rcond'] = None  # explicitly set future default
    return _original_lstsq(*args, **kwargs)


np.linalg.lstsq = patched_lstsq
