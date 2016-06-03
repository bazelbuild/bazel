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

def _parent_dirs(dirs):
  """Returns a set of parent directories for each directory in dirs."""
  return set([f.rpartition("/")[0] for f in dirs])

def _intersperse(separator, iterable):
  """Inserts separator before each item in iterable."""
  result = []
  for x in iterable:
    result.append(separator)
    result.append(x)

  return result

def _swift_target(cpu, sdk_version):
  """Returns a target triplet for Swift compiler."""
  return "%s-apple-ios%s" % (cpu, sdk_version)

def _module_name(ctx):
  """Returns a module name for the given rule context."""
  return ctx.label.package.lstrip("//").replace("/", "_") + "_" + ctx.label.name

def _swift_library_impl(ctx):
  """Implementation for swift_library Skylark rule."""
  cpu = ctx.fragments.apple.ios_cpu()
  platform = ctx.fragments.apple.ios_cpu_platform()
  sdk_version = ctx.fragments.apple.sdk_version_for_platform(platform)
  target = _swift_target(cpu, sdk_version)
  apple_toolchain = apple_common.apple_toolchain()

  module_name = ctx.attr.module_name or _module_name(ctx)

  # A list of paths to pass with -F flag.
  framework_dirs = set([
      apple_toolchain.platform_developer_framework_dir(ctx.fragments.apple)])

  # Collect transitive dependecies.
  dep_modules = []
  dep_libs = []

  swift_providers = [x.swift for x in ctx.attr.deps if hasattr(x, "swift")]
  objc_providers = [x.objc for x in ctx.attr.deps if hasattr(x, "objc")]

  for swift in swift_providers:
    dep_libs += swift.transitive_libs
    dep_modules += swift.transitive_modules

  objc_includes = set()    # Everything that needs to be included with -I
  objc_files = set()       # All inputs required for the compile action
  for objc in objc_providers:
    objc_includes += objc.include
    objc_includes = objc_includes.union([x.dirname for x in objc.module_map])

    objc_files += objc.header
    objc_files += objc.module_map

    files = set(objc.static_framework_file) + set(objc.dynamic_framework_file)
    objc_files += files
    framework_dirs += _parent_dirs(objc.framework_dir)

  # TODO(b/28005753): Currently this is not really a library, but an object
  # file, does not matter to the linker, but should be replaced with proper ar
  # call.
  output_lib = ctx.new_file(module_name + ".a")
  output_module = ctx.new_file(module_name + ".swiftmodule")
  output_header = ctx.new_file(ctx.label.name + "-Swift.h")

  srcs_args = [f.path for f in ctx.files.srcs]

  # TODO(b/28005582): Instead of including a dir for each dependecy, output to
  # a shared dir and include that?
  include_dirs = set([x.dirname for x in dep_modules])

  include_args = ["-I%s" % d for d in include_dirs + objc_includes]
  framework_args = ["-F%s" % x for x in framework_dirs]

  # Add the current directory to clang's search path.
  # This instance of clang is spawned by swiftc to compile module maps and is
  # not passed the current directory as a search path by default.
  clang_args = _intersperse("-Xcc", ["-iquote", "."])

  args = [
      "swift",
      "-frontend",
      "-emit-object",
      "-emit-module-path", output_module.path,
      "-module-name", module_name,
      "-emit-objc-header-path", output_header.path,
      "-parse-as-library",
      "-target", target,
      "-sdk", apple_toolchain.sdk_dir(),
      "-o", output_lib.path,
      ] + srcs_args + include_args + framework_args + clang_args

  xcrun_action(
      ctx,
      inputs = ctx.files.srcs + dep_modules + dep_libs + list(objc_files),
      outputs = (output_lib, output_module, output_header),
      mnemonic = "SwiftCompile",
      arguments = args,
      use_default_shell_env = False,
      progress_message = ("Compiling Swift module %s (%d files)"
                          % (ctx.label.name, len(ctx.files.srcs))))

  objc_provider = apple_common.new_objc_provider(
      library=set([output_lib] + dep_libs),
      header=set([output_header]),
      providers=objc_providers,
      uses_swift=True)

  return struct(
      swift=struct(
          transitive_libs=[output_lib] + dep_libs,
          transitive_modules=[output_module] + dep_modules),
      objc=objc_provider,
      files=set([output_lib, output_module, output_header]))

swift_library = rule(
    _swift_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = FileType([".swift"])),
        "deps": attr.label_list(providers=[["swift"], ["objc"]]),
        "module_name": attr.string(mandatory=False),
        "_xcrunwrapper": attr.label(
            executable=True,
            default=Label(XCRUNWRAPPER_LABEL))},
    fragments = ["apple"],
    output_to_genfiles=True,
)
"""
Builds a Swift module.

A module is a pair of static library (.a) + module header (.swiftmodule).
Dependant targets can import this module as "import RuleName".

Args:
  srcs: Swift sources that comprise this module.
  deps: Other Swift modules.
  module_name: Optional. Sets the Swift module name for this target. By default
               the module name is the target path with all special symbols
               replaced by "_", e.g. //foo:bar can be imported as "foo_bar".
"""
