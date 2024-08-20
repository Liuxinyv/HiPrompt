from pathlib import Path
from PIL import Image
import numpy as np

from .identity import IdentityView
from .hybrid import HybridLowPassView, HybridHighPassView
VIEW_MAP = {
    'identity': IdentityView,
    'low_pass': HybridLowPassView,
    'high_pass': HybridHighPassView,

}

def get_views(view_names, view_args=None,beta=None):
    '''
    Bespoke function to get views (just to make command line usage easier)
    '''

    views = []
    if view_args is None:
        view_args = [None for _ in view_names]

    for view_name, view_arg in zip(view_names, view_args):
        if view_name in ['low_pass']:
            args = [2.0 if view_arg is None else float(view_arg)]
        elif view_name in ['high_pass']:
            args = [2.0 if view_arg is None else float(view_arg), beta]
        else:
            args = []
        print(args)
        view = VIEW_MAP[view_name](*args)
        views.append(view)
    
    return views
