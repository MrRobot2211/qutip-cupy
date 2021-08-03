"""Contains specialization functions for dense_cupy. These are the functions that
 are defined outside of qutip/core/data/dense.pyx."""

import cupy as cp

from .dense import CuPyDense


def tidyup_dense(matrix, tol, inplace=True):
    return matrix


def reshape_cupydense(cp_arr, n_rows_out, n_cols_out):

    return CuPyDense._raw_cupy_constructor(
        cp.reshape(cp_arr._cp, (n_rows_out, n_cols_out))
    )


def _check_square_matrix(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "".join(["matrix shape ", str(matrix.shape), " is not square."])
        )


def trace_cupydense(cp_arr):
    _check_square_matrix(cp_arr)
    # @TODO: whnen qutip allows it we should remove this call to item()
    # as it takes a time penalty commmunicating data from GPU to CPU.
    return cp.trace(cp_arr._cp).item()


def _check_ptrace_shape(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ptrace is only defined for square density matrices")


def _prepare_inputs(dims, sel):

    dims = cp.atleast_1d(dims).ravel()
    sel = cp.atleast_1d(sel)
    if sel.ndim != 1:
        raise ValueError("Selection must be one-dimensional")
    sel.sort()
    for i in range(sel.shape[0]):
        if sel[i] < 0 or sel[i] >= dims.size:
            raise IndexError("Invalid selection index in ptrace.")
        if i > 0 and sel[i] == sel[i - 1]:
            raise ValueError("Duplicate selection index in ptrace.")
    return dims, sel


def ptrace_dense(matrix, dims, sel):
    _check_ptrace_shape(matrix)
    dims, sel = _prepare_inputs(dims, sel)
    if len(sel) == len(dims):
        return matrix.copy()
    nd = dims.shape[0]
    dkeep = [dims[x] for x in sel]
    qtrace = list(set(cp.arange(nd)) - set(sel))
    dtrace = [dims[x] for x in qtrace]
    dims = list(dims)
    sel = list(sel)

    # This could be accomplished alternatively by doing the reshape dims+ dims
    # calling eiunsum on the unselected axis
    rhomat = cp.trace(
        matrix._cp.reshape(dims + dims)
        .transpose(qtrace + [nd + q for q in qtrace] + sel + [nd + q for q in sel])
        .reshape([cp.prod(dtrace), cp.prod(dtrace), cp.prod(dkeep), cp.prod(dkeep)])
    )
    return CuPyDense._raw_cupy_constructor(rhomat)
