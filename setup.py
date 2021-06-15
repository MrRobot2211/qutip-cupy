#!/usr/bin/env python

import os
import subprocess
import sys

# Required third-party imports, must be specified in pyproject.toml.
import packaging.version
import setuptools


def process_options():
    """
    Determine all runtime options, returning a dictionary of the results.  The
    keys are:
        'rootdir': str
            The root directory of the setup.  Almost certainly the directory
            that this setup.py file is contained in.
        'release': bool
            Is this a release build (True) or a local development build (False)
    """
    options = {}
    options['rootdir'] = os.path.dirname(os.path.abspath(__file__))
    options = _determine_version(options)
    return options


def _determine_version(options):
    """
    Adds the 'short_version', 'version' and 'release' options.

    Read from the VERSION file to discover the version.  This should be a
    single line file containing valid Python package public identifier (see PEP
    440), for example
      4.5.2rc2
      5.0.0
      5.1.1a1
    We do that here rather than in setup.cfg so we can apply the local
    versioning number as well.
    """
    version_filename = os.path.join(options['rootdir'], 'VERSION')
    with open(version_filename, "r") as version_file:
        version_string = version_file.read().strip()
    version = packaging.version.parse(version_string)
    if isinstance(version, packaging.version.LegacyVersion):
        raise ValueError("invalid version: " + version_string)
    options['short_version'] = str(version.public)
    options['release'] = not version.is_devrelease
    if not options['release']:
        # Put the version string into canonical form, if it wasn't already.
        version_string = str(version)
        version_string += "+"
        try:
            git_out = subprocess.run(
                ('git', 'rev-parse', '--verify', '--short=7', 'HEAD'),
                check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            git_hash = git_out.stdout.decode(sys.stdout.encoding).strip()
            version_string += git_hash or "nogit"
        # CalledProcessError is for if the git command fails for internal
        # reasons (e.g. we're not in a git repository), OSError is for if
        # something goes wrong when trying to run git (e.g. it's not installed,
        # or a permission error).
        except (subprocess.CalledProcessError, OSError):
            version_string += "nogit"
    options['version'] = version_string
    return options


def create_version_py_file(options):
    """
    Generate and write out the file version.py, which is used to produce the
    '__version__' information for the module.  This function will overwrite an
    existing file at that location.
    """
    filename = os.path.join(
        options['rootdir'], 'src', 'qutip_cupy', 'version.py',
    )
    content = "\n".join([
        "# This file is automatically generated during package setup.",
        f"short_version = '{options['short_version']}'",
        f"version = '{options['version']}'",
        f"release = {options['release']}",
    ])
    with open(filename, 'w') as file:
        print(content, file=file)

#The following is required to get unit tests up and running.
# If the user doesn't have, then that's OK, we'll just skip unit tests.
try:
    from setuptools import setup, Extension
    EXTRA_KWARGS = {
        'tests_require': ['pytest']
    }
except:
    from distutils.core import setup
    from distutils.extension import Extension
    EXTRA_KWARGS = {}

from Cython.Build import cythonize
from Cython.Distutils import build_ext

# Cython extensions to be compiled.  The key is the relative package name, the
# value is a list of the Cython modules in that package.
cy_exts = {
    'qutip_cupy': [
        'cupy_dense']}

# Extra link args
_link_flags = []

# If on Win and Python version >= 3.5 and not in MSYS2
# (i.e. Visual studio compile)
if (
    sys.platform == 'win32'
    and int(str(sys.version_info[0])+str(sys.version_info[1])) >= 35
    and os.environ.get('MSYSTEM') is None
):
    _compiler_flags = ['/w', '/Ox']
# Everything else
else:
    _compiler_flags = ['-w', '-O3', '-funroll-loops']
    if sys.platform == 'darwin':
        # These are needed for compiling on OSX 10.14+
        _compiler_flags.append('-mmacosx-version-min=10.9')
        _link_flags.append('-mmacosx-version-min=10.9')
import numpy as np
EXT_MODULES = []
_include = [
    np.get_include(),
]

# Add Cython files from qutip
for package, files in cy_exts.items():
    for file in files:
        _module = 'src' + ('.' + package if package else '') + '.' + file
        _file = os.path.join('src', *package.split("."), file + '.pyx')
        _sources = [_file, 'qutip/core/data/src/matmul_csr_vector.cpp']
        EXT_MODULES.append(Extension(_module,
                                     sources=_sources,
                                     include_dirs=_include,
                                     extra_compile_args=_compiler_flags,
                                     extra_link_args=_link_flags,
                                     language='c++'))

# Remove -Wstrict-prototypes from cflags
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
if "CFLAGS" in cfg_vars:
    cfg_vars["CFLAGS"] = cfg_vars["CFLAGS"].replace("-Wstrict-prototypes", "")

if __name__ == "__main__":
    options = process_options()
    create_version_py_file(options)
    # Most of the kwargs to setup are defined in setup.cfg; the only ones we
    # keep here are ones that we have done some compile-time processing on.
    setuptools.setup(
        version=options['version'],
        ext_modules=cythonize(EXT_MODULES),
        cmdclass={'build_ext': build_ext}
    )
