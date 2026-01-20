import numpy as np
import logging

def should_stop(y, depth, max_depth, min_samples):
    if len(np.unique(y)) == 1:
        logging.info("Stopping: pure node")
        return True

    if depth >= max_depth:
        logging.info("Stopping: max depth reached")
        return True

    if len(y) < min_samples:
        logging.info("Stopping: not enough samples")
        return True

    return False