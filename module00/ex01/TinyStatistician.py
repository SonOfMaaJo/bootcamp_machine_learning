from math import floor, sqrt, ceil
import numpy as np


class TinyStatistician():
    """Class to perform statistics values on iterator data"""
    def mean(self, x):
        """
        function to calcul the mean of a values of list or numpy.array
        Args:
        _____
            x: list or np.array of data
        Return:
        _____
            mean: mean of the data
        """
        if not isinstance(x, (list, np.array)) or len(x) == 0:
            return None
        sums = 0
        for el in x:
            sums = sums + el
        return sums / len(x)

    @staticmethod
    def median(x):
        """function to calcul the mediann of a values of list or
        numpy.array
        Args:
        _____
            x : list or np.array
        Retrun:
        _____
            median : float representing the median of the data
        """
        if not isinstance(x, (list, np.array)) or len(x) == 0:
            return None
        data = sorted(x)
        return float(data[floor(len(data) / 2)])

    @staticmethod
    def quartile(x):
        """Method to calcul the first and third quatile of the data
        Args:
        ______
            x:  list or np.array of the data.
        Return:
        ______
            lst:    list or the first and third quatile of the data
        """
        if not isinstance(x, (list, np.array)) or len(x) == 0:
            return None
        data = sorted(x)
        ln = len(data)
        return  [float(data[floor(ln / 4)]),
                 float(data[floor(3 * ln / 4)])]

    @staticmethod
    def percentile(x, p):
        """Method to computes the expected percentile of a giben non-empty
        list or array x.
        Args:
        _____
            x:  list or np.array of data.
            p:  demanded percentile
        Returns:
        ______
            percentile: float representing the expected percentile.
        """
        if not isinstance(x, (list, np.array)) or len(x) == 0:
            return None
        data = sorted(x)
        pos = (p * (len(x) + 1)) / 100
        if pos.is_integer():
            return float(data[pos])
        return float((data[floor(pos)] + data[ceil(pos)]) / 2)

    def var(self, x):
        """Method to calcul variance of a given dataset x.
        Args:
        _____
            x:  list or np.array of a data
        Return:
        _____
            sigma_square: variance of the datas
        """
        moy = self.mean(x)
        if moy is None:
            return None
        sigma_square = 0
        for el in x:
            sigma_square += float(el - moy)**2
        return sigma_square / float(len(x))

    def std(self, x):
        """Calcul the standard deviation of a given dataset.
        Args:
        _____
            x:  list or np.array of data
        Return:
            sigma:  the standard deviation of a given data
        """
        sigma_square = self.var(x)
        return sqrt(sigma_square) if sigma_square else None
