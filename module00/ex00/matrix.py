class Matrix():
    """Class to implement and manage matrix operations"""
    def __init__(self, entry):
        if isinstance(entry, tuple):
            self.data = [[0 for i in range(entry[1])] for i in
                         range(entry[0])]
            self.shape = entry
        elif isinstance(entry, list):
            lens = [len(ent) for ent in entry]
            if len(set(lens)) != 1:
                raise ValueError("Imcompatible dimensions in the elements.")
            self.data = entry
            self.shape = (len(entry), len(entry[0]))
        else:
            raise TypeError("The entry should be a list of list of real"
                            "numbers , or a tuple (int, int).")

    def __add__(self, other):
        if not isinstance(other, type(self)) or self.shape != other.shape:
            raise NotImplementedError("Couldn't perform this operations")
        return Matrix([[self.data[i][j] + other.data[i][j] for j in
                        range(self.shape[1])] for i in
                       range(self.shape[0])])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, type(self)) or self.shape != other.shape():
            raise NotImplementedError("Couldn't perform this operations")
        return Matrix([[self.data[i][j] - other.data[i][j] for j in
                        range(self.shape[1])] for i in
                       range(self.shape[0])])

    def __rsub__(self, other):
        return other.__sub__(self)

    def __truediv__(self, scalar):
        if not isinstance(scalar, float) and not isinstance(scalar, int):
            raise TypeError("The division is perform with int or float")
        return Matrix([[self.data[i][j] / scalar for j in
                        range(self.shape[1])] for i in
                       range(self.shape[0])])

    def __rtruediv__(self, scalar):
        raise NotImplementedError("The operations is not possible.")

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Matrix([[self.data[i][j] * other for j in
                            range(self.shape[1])] for i in
                           range(self.shape[0])])
        if isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise NotImplementedError("This operations is not possible")
            data = [[sum([self.data[i][k] * other.data[k][j]
                          for k in range(self.shape[1])])
                     for j in range(other.shape[1])]
                    for i in range(self.shape[0])]
            if self.shape[0] == 1 or other.shape[1] == 1:
                return Vector(data)
            return Matrix(data)
        raise NotImplementedError("This operations is not possible")

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        if isinstance(other, Matrix):
            return other.__mul__(self)
        raise NotImplementedError("The operations is not possible")

    def __str__(self):
        return f"{type(self).__name__}({self.data})"

    def __repr__(self):
        return f"{type(self).__name__}({self.data})"

    def T(self):
        """function to perform transpose of Matrix.
        Args:
            Matrix: a matrix.
        Returns:
            other: The transpose of a Matrix.
        """
        data = [[self.data[i][j] for i in range(self.shape[0])]
                for j in range(self.shape[1])]
        return Matrix(data)

class Vector(Matrix):
    """Class to manage vectors and operations over them."""
    def __init__(self, entry):
        if not isinstance(entry, list):
            raise NotImplementedError("The entry should be a list.")
        if len(entry) != 1:
            lens = [ent for ent in entry if len(ent) != 1]
            if len(lens) != 0:
                raise NotImplementedError("Format error for vector type")
        super().__init__(entry)

    def __add__(self, other):
        result = super().__add__(other)
        return Vector(result.data)
    def __radd__(self, other):
        result = super().__radd__(other)
        return Vector(result.data)

    def __sub__(self, other):
        result = super().__sub__(other)
        return Vector(result.data)

    def __rsub__(self, other):
        return other.__sub__(self)


    def dot(self, v):
        """Method to perfom scalar product between two vector.
        Args:
            v: a vector instance.
        Returns:
            res: a float type representing the result of scalar product.
        """
        if not isinstance(v, Vector):
            raise NotImplementedError("Imcompatible operations")
        if v.shape != self.shape:
            raise NotImplementedError("Differents shapes")
        if self.shape[0] == 1:
            return sum([self.data[0][i] * v.data[0][i]
                        for i in range(self.shape[1])])
        return sum([self.data[i][0] * v.data[i][0]
                    for i in range(self.shape[0])])
