load(":python_version.bzl", "define_python_version_flag")

package(default_visibility = ["//visibility:public"])

sh_binary(
    name = "2to3",
    srcs = ["2to3.sh"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/python/runfiles:srcs",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["**"]) + [
        "//tools/python/runfiles:embedded_tools",
    ],
    visibility = ["//tools:__pkg__"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        "//tools/python/runfiles:all_windows_tests",
    ],
    visibility = ["//tools:__pkg__"],
)

# This target can be used to inspect the current Python major version. To use,
# put it in the `flag_values` attribute of a `config_setting` and test it
# against the values "PY2" or "PY3". It will always match one or the other.
#
# If you do not need to test any other flags in combination with the Python
# version, then as a convenience you may use the predefined `config_setting`s
# `@bazel_tools//tools/python:PY2` and `@bazel_tools//tools/python:PY3`.
#
# Example usage:
#
#     config_setting(
#         name = "py3_on_arm",
#         values = {"cpu": "arm"},
#         flag_values = {"@bazel_tools//tools/python:python_version": "PY3"},
#     )
#
#     my_target(
#         ...
#         some_attr = select({
#             ":py3_on_arm": ...,
#             ...
#         }),
#         ...
#     )
#
# Caution: Do not `select()` on the built-in command-line flags `--force_python`
# or `--python_version`, as they do not always reflect the true Python version
# of the current target. `select()`-ing on them can lead to action conflicts and
# will be disallowed.
define_python_version_flag(
    name = "python_version",
)

config_setting(
    name = "PY2",
    flag_values = {":python_version": "PY2"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "PY3",
    flag_values = {":python_version": "PY3"},
    visibility = ["//visibility:public"],
)

# The toolchain type for Python rules. Provides a Python 2 and/or Python 3
# runtime.
toolchain_type(name = "toolchain_type")

# A constraint_setting to use for constraints related to the location of the
# system Python 2 interpreter on a platform.
constraint_setting(name = "py2_interpreter_path")

# A constraint_setting to use for constraints related to the location of the
# system Python 3 interpreter on a platform.
constraint_setting(name = "py3_interpreter_path")
