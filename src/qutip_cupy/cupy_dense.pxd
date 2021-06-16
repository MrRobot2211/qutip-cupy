#cython: language_level=3

cimport numpy as cnp

#from . cimport base
#from .csr cimport CSR

#from qutip cimport data

ctypedef cnp.npy_int32 idxint
cdef int idxint_DTYPE

cdef class Dense:
#    cdef double complex *data
#    cdef bint fortran
    cdef public object cpa
    cdef readonly (idxint, idxint) shape
    # cdef bint _deallocate
    # cdef void _fix_flags(Dense self, object array, bint make_owner=*)
    # cpdef Dense reorder(Dense self, int fortran=*)
    # cpdef Dense copy(Dense self)
    # cpdef object as_ndarray(Dense self)
    # cpdef object to_array(Dense self)
    # cpdef double complex trace(Dense self)
    # cpdef Dense adjoint(Dense self)
    # cpdef Dense conj(Dense self)
    # cpdef Dense transpose(Dense self)

# cpdef Dense fast_from_numpy(object array)
# cdef Dense wrap(double complex *ptr, base.idxint rows, base.idxint cols, bint fortran=*)
# cpdef Dense empty(base.idxint rows, base.idxint cols, bint fortran=*)
# cpdef Dense empty_like(Dense other, int fortran=*)
# cpdef Dense zeros(base.idxint rows, base.idxint cols, bint fortran=*)
# cpdef Dense identity(base.idxint dimension, double complex scale=*,
#                      bint fortran=*)
# cpdef Dense from_csr(CSR matrix, bint fortran=*)