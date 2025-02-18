"""
This module contains the ``CuPyDense`` class and associated function
conversion and specializations for registration with QuTiP's data layer.
"""

import numbers

import cupy as cp
from qutip.core import data


class CuPyDense(data.Data):
    """
    This class provides a dense matrix backend for QuTiP.
    Matrices are stored internally in a CuPy array on a GPU.
    If you have many GPUs you can set GPU ``i``
    by calling ``cp.cuda.Device(i).use()`` before construction.

    Parameters
    ----------
    data: array-like
        Data to be stored.
    shape: (int, int)
        Defaults to ``None``. If ``None`` will infer the shape from ``data``,
        else it will set the shape for the internal CuPy array.
    copy: bool
        Defaults to ``True``. Whether to make a copy of
        the elements in ``data`` or not.
    dtype:
        Data type specifier. Either ``cp.complex128`` or ``cp.complex64``
    """

    def __init__(self, data, shape=None, copy=True, dtype=cp.complex128):
        self.dtype = dtype
        base = cp.array(data, dtype=self.dtype, order="K", copy=copy)
        if shape is None:
            shape = base.shape
            # Promote to a ket by default if passed 1D data.
            if len(shape) == 1:
                shape = (shape[0], 1)
        if not (
            len(shape) == 2
            and isinstance(shape[0], numbers.Integral)
            and isinstance(shape[1], numbers.Integral)
            and shape[0] > 0
            and shape[1] > 0
        ):
            raise ValueError(
                f"shape must be a 2-tuple of positive ints, but is {shape!r}"
            )
        if shape and (shape[0] != base.shape[0] or shape[1] != base.shape[1]):
            if shape[0] * shape[1] != base.size:
                raise ValueError(
                    f"invalid shape {shape} for input data with size {base.shape}"
                )
            else:
                self._cp = base.reshape(shape)
        else:
            self._cp = base

        super().__init__((shape[0], shape[1]))

    @classmethod
    def _raw_cupy_constructor(cls, data):
        """
        A fast low-level constructor for wrapping an existing CuPy array in a
        CuPyDense object without copying it.

        The ``data`` argument must be a CuPy array with the correct shape.
        The CuPy array will not be copied and will be used as is.
        """
        out = cls.__new__(cls)
        super(cls, out).__init__(data.shape)
        out._cp = data
        out.dtype = data.dtype
        return out

    def copy(self):
        return self._raw_cupy_constructor(self._cp.copy())

    def to_array(self):
        return cp.asnumpy(self._cp)

    def conj(self):
        return CuPyDense._raw_cupy_constructor(self._cp.conj())

    def transpose(self):
        return CuPyDense._raw_cupy_constructor(self._cp.transpose())

    def adjoint(self):
        return CuPyDense._raw_cupy_constructor(self._cp.transpose().conj())

    def trace(self):
        return self._cp.trace()


# @TOCHECK I added docstrings describing functions as they are.
# If we were to have a precision parameter on the conversion
# I am not really sure how the dispatcher would handle it.
# It looks like we may be needing 2 classes.
def dense_from_cupydense(cupydense):
    """
    Creates a QuTiP ``data.Dense`` array from the values in a CuPyDense array.
    The resulting array has complex128 precision.
    """
    dense_np = data.Dense(cupydense.to_array(), copy=False)
    return dense_np


def cupydense_from_dense(dense):
    """
    Creates a CuPyDense array from the values in a QuTiP ``data.Dense`` array
    with ``cp.complex128`` precision.
    """
    dense_cp = CuPyDense(dense.as_ndarray(), copy=False)
    return dense_cp


def cpd_adjoint(cpd_array):
    return cpd_array.adjoint()


def cpd_conj(cpd_array):
    return cpd_array.conj()


def cpd_transpose(cpd_array):
    return cpd_array.transpose()


def cpd_trace(cpd_array):
    return cpd_array.trace()
