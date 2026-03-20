# utils/seed.py

import os
import random
import numpy as np


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds across all frameworks for reproducibility."""
    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
