# gflags raises DuplicateFlagError when defining default flags from packages
# with different names, so this pseudo-package must mimic the core gflags
# package name.
__name__ += ".gflags"  # i.e. "third_party.py.gflags.gflags"

from gflags import *
