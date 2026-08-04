import numpy as np


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if y.ndim == 2 and y.shape[1] == 1 and y.shape == y_hat.shape:
        return float(y_hat[y_hat == y].size / y.size)
    return None


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int,
        the class on which to report the precision_score (default=1)
    Returns:
        The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if not isinstance(pos_label, (str, int)):
        return None
    if y.ndim == 2 and y.shape[1] == 1 and y.shape == y_hat.shape:
        tp = y[(y == pos_label) & (y_hat == pos_label)].size
        fp = y[(y != pos_label) & (y_hat == pos_label)].size
        return 1 / (1 + fp / tp) if tp != 0 else None
    return None


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to
        report the precision_score (default=1)
    Returns:
        The recall score as a float.
        None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if not isinstance(pos_label, (str, int)):
        return None
    if y.ndim == 2 and y.shape[1] == 1 and y.shape == y_hat.shape:
        tp = y[(y == pos_label) & (y_hat == pos_label)].size
        fn = y[(y == pos_label) & (y_hat != pos_label)].size
        return 1 / (1 + fn / tp) if tp != 0 else None
    return None


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which
        to report the precision_score (default=1)
    Returns:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    return 2 * precision * recall / (precision + recall) if precision is not\
        None and recall is not None else None
