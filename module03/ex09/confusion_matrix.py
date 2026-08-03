import numpy as np
import pandas as pd


def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true: a numpy.ndarray for the correct labels
        y_hat: a numpy.ndarray for the predicted labels
        labels: optional, a list of labels to index the matrix.
        This may be used to reorder or select a subset of labels.(default=None)
        df_option: optional, if set to True the function will return
        a pandas DataFrame instead of a numpy array. (default=False)
    Returns:
        Confusion matrix as a numpy ndarray
        or a pandas DataFrame according to df_option value.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y_true, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    if labels is not None and not isinstance(labels, list):
        return None
    if labels is None:
        labels = list(np.unique(y_hat))
    confus_matrix = np.array([[y_true[(y_true == label_row) &
                                      (y_hat == label_col)
                                      ].size for label_col in labels]
                              for label_row in labels])
    return pd.DataFrame(confus_matrix, index=labels, columns=labels) \
        if df_option else confus_matrix
