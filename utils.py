import numpy as np

def quadratic_error_propagation(data: np.ndarray, error: np.ndarray):
    """Calculates the standard deviation when taking the square root
    of the squared visibilities."""
    return np.abs(0.5*error/np.sqrt(data))
