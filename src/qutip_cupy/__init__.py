""" The qutip-cupy package provides a CuPy-based data layer for QuTiP. """

# we need to silence this specific warning
# remember to remove once QuTiP moves matplotlib
# to an official optional dependency
import warnings

try:
    __import__("cupy")
except ModuleNotFoundError:
    raise RuntimeError(
        "qutip_cupy requires cupy to be installed, please install cupy by following "
        "the instructions at https://docs.cupy.dev/en/stable/install.html"
    )

with warnings.catch_warnings():
    warnings.filterwarnings(
        action="ignore", category=UserWarning, message=r"matplotlib not found:"
    )

    from qutip.core import data

# qutip_cupy imports need to be after the cupy import check above
from .version import version as __version__  # noqa: E402
from . import dense as cd  # noqa: E402

__all__ = ["__version__", "CuPyDense"]

CuPyDense = cd.CuPyDense

data.to.add_conversions(
    [
        (CuPyDense, data.Dense, cd.cupydense_from_dense),
        (data.Dense, CuPyDense, cd.dense_from_cupydense),
    ]
)
data.to.register_aliases(["cupyd"], CuPyDense)

data.adjoint.add_specialisations([(CuPyDense, CuPyDense, cd.cpd_adjoint)])
data.transpose.add_specialisations([(CuPyDense, CuPyDense, cd.cpd_transpose)])
data.conj.add_specialisations([(CuPyDense, CuPyDense, cd.cpd_conj)])
data.trace.add_specialisations([(CuPyDense, cd.cpd_trace)])

# We must register the functions to the data layer but do not want
# the data layer or qutip_cupy.dense to be accessible from qutip_cupy
del data
del cd
