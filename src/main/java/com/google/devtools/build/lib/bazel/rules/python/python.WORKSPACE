# Don't use the target pattern :all, because we don't want to register
# :autodetecting_toolchain_nonstrict. That toolchain opts out of Python version
# checking, so the user should have to explicitly declare it on their command
# line or WORKSPACE file. (And even if we did register it here, we still
# couldn't use :all because we'd want to ensure it has the lowest possible
# priority.)
register_toolchains("@bazel_tools//tools/python:autodetecting_toolchain")
