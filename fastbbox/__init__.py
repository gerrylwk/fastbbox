from .bbox import bbox_overlaps, generalized_iou, distance_iou, complete_iou, efficient_iou, normalized_wasserstein_distance
from .obb_bbox import bbox_overlaps_obb

# Detect which backend is being used
try:
    from . import bbox as _bbox_module
    # Check the module's docstring which identifies the backend
    if _bbox_module.__doc__ and 'nanobind' in _bbox_module.__doc__:
        _backend = 'nanobind'
    elif _bbox_module.__doc__ and ('cython' in _bbox_module.__doc__.lower() or 'pyx' in str(_bbox_module.__file__)):
        _backend = 'cython'
    else:
        # Fallback: check file extension and patterns
        if hasattr(_bbox_module, '__file__') and _bbox_module.__file__:
            file_str = str(_bbox_module.__file__)
            if '_nb' in file_str or 'nanobind' in file_str:
                _backend = 'nanobind'
            else:
                _backend = 'cython'
        else:
            _backend = 'unknown'
except Exception:
    _backend = 'unknown'

__version__ = "0.1.0"
__backend__ = _backend
__all__ = [
    "bbox_overlaps", "generalized_iou", "distance_iou", "complete_iou", 
    "efficient_iou", "normalized_wasserstein_distance",
    "bbox_overlaps_obb",
    "__backend__"
]


