# Copyright 2018 Google Inc. All Rights Reserved.
#
# Distributed under MIT license.
#  See file LICENSE for detail or copy at https://opensource.org/licenses/MIT

"""Creates config_setting that allows selecting based on 'compiler' value."""

def create_msvc_config():
  # The "do_not_use_tools_cpp_compiler_present" attribute exists to
  # distinguish between older versions of Bazel that do not support
  # "@bazel_tools//tools/cpp:compiler" flag_value, and newer ones that do.
  # In the future, the only way to select on the compiler will be through
  # flag_values{"@bazel_tools//tools/cpp:compiler"} and the else branch can
  # be removed.
  if hasattr(cc_common, "do_not_use_tools_cpp_compiler_present"):
    native.config_setting(
      name = "msvc",
      flag_values = {
          "@bazel_tools//tools/cpp:compiler": "msvc-cl",
      },
      visibility = ["//visibility:public"],
    )
  else:
    native.config_setting(
      name = "msvc",
      values = {"compiler": "msvc-cl"},
      visibility = ["//visibility:public"],
    )
