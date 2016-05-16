# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Skylark rules for Swift."""

load("shared", "xcrun_action", "XCRUNWRAPPER_LABEL")

def _swift_target(cpu, sdk_version):
  """Returns a target triplet for Swift compiler."""
  return "%s-apple-ios%s" % (cpu, sdk_version)

def _swift_library_impl(ctx):
  """Implementation for swift_library Skylark rule."""
  cpu = ctx.fragments.apple.ios_cpu()
  platform = ctx.fragments.apple.ios_cpu_platform()
  sdk_version = ctx.fragments.apple.sdk_version_for_platform(platform)
  target = _swift_target(cpu, sdk_version)

  # Collect transitive dependecies.
  dep_modules = []
  dep_libs = []
  for x in ctx.attr.deps:
    swift_provider = x.swift
    dep_libs.append(swift_provider.library)
    dep_libs += swift_provider.transitive_libs

    dep_modules.append(swift_provider.module)
    dep_modules += swift_provider.transitive_modules

  # TODO(b/28005753): Currently this is not really a library, but an object
  # file, does not matter to the linker, but should be replaced with proper ar
  # call.
  output_lib = ctx.outputs.swift_lib
  output_module = ctx.outputs.swift_module
  output_header = ctx.outputs.swift_header

  srcs_args = [f.path for f in ctx.files.srcs]

  # TODO(b/28005582): Instead of including a dir for each dependecy, output to
  # a shared dir and include that?
  include_dirs = set([x.dirname for x in dep_modules])
  include_args = ["-I%s" % d for d in include_dirs]

  args = [
      "swift",
      "-frontend",
      "-emit-object",
      "-emit-module-path", output_module.path,
      "-module-name", ctx.label.name,
      "-emit-objc-header-path", output_header.path,
      "-parse-as-library",
      "-target", target,
      "-sdk", apple_common.apple_toolchain().sdk_dir(),
      "-o", output_lib.path,
      ] + srcs_args + include_args

  xcrun_action(
      ctx,
      inputs = ctx.files.srcs + dep_modules + dep_libs,
      outputs = (output_lib, output_module, output_header),
      mnemonic = "SwiftCompile",
      arguments = args,
      use_default_shell_env = False,
      progress_message = ("Compiling Swift module %s (%d files)"
                          % (ctx.label.name, len(ctx.files.srcs))))

  struct_kw = {}
  if hasattr(apple_common, "new_objc_provider"):
    struct_kw["objc"] = apple_common.new_objc_provider(
        library=set([output_lib] + dep_libs),
        header=set([output_header]))
  else:
    # TODO(cl/121390911): Remove when this is released.
    struct_kw["objc_export"] = struct(library=set([output_lib] + dep_libs),
                                      header=set([output_header]))

  return struct(
      swift=struct(
          library=output_lib,
          module=output_module,
          transitive_libs=dep_libs,
          transitive_modules=dep_modules), **struct_kw)

swift_library = rule(
    _swift_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = FileType([".swift"])),
        "deps": attr.label_list(providers=["swift"]),
        "_xcrunwrapper": attr.label(
            executable=True,
            default=Label(XCRUNWRAPPER_LABEL))},
    fragments = ["apple"],
    output_to_genfiles=True,
    outputs = {
        "swift_lib": "%{name}.a",
        "swift_module": "%{name}.swiftmodule",
        "swift_header": "%{name}-Swift.h",
    },
)
"""
Builds a Swift module.

A module is a pair of static library (.a) + module header (.swiftmodule).
Dependant targets can import this module as "import RuleName".

Args:
  srcs: Swift sources that comprise this module.
  deps: Other Swift modules.
"""
