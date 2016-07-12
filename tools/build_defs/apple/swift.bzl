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

load("shared", "xcrun_action", "XCRUNWRAPPER_LABEL", "module_cache_path")

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

def _swift_compilation_mode_flags(ctx):
  """Returns additional swiftc flags for the current compilation mode."""
  mode = ctx.var["COMPILATION_MODE"]
  if mode == "dbg":
    return ["-Onone", "-DDEBUG", "-g", "-enable-testing"]
  elif mode == "fastbuild":
    return ["-Onone", "-DDEBUG", "-enable-testing"]
  elif mode == "opt":
    return ["-O", "-DNDEBUG"]

def _clang_compilation_mode_flags(ctx):
  """Returns additional clang flags for the current compilation mode."""

  # In general, every compilation mode flag from native objc_ rules should be
  # passed, but -g seems to break Clang module compilation. Since this flag does
  # not make much sense for module compilation and only touches headers,
  # it's ok to omit.
  native_clang_flags = ctx.fragments.objc.copts_for_current_compilation_mode

  return [x for x in native_clang_flags if x != "-g"]

def _module_name(ctx):
  """Returns a module name for the given rule context."""
  return ctx.label.package.lstrip("//").replace("/", "_") + "_" + ctx.label.name

def _swift_library_impl(ctx):
  """Implementation for swift_library Skylark rule."""
  # TODO(b/29772303): Assert xcode version.
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

  objc_includes = set()     # Everything that needs to be included with -I
  objc_files = set()        # All inputs required for the compile action
  objc_module_maps = set()  # Module maps for dependent targets
  objc_defines = set()
  for objc in objc_providers:
    objc_includes += objc.include

    objc_files += objc.header
    objc_files += objc.module_map

    objc_module_maps += objc.module_map

    files = set(objc.static_framework_file) + set(objc.dynamic_framework_file)
    objc_files += files
    framework_dirs += _parent_dirs(objc.framework_dir)

    # objc_library#copts is not propagated to its dependencies and so it is not
    # collected here. In theory this may lead to un-importable targets (since
    # their module cannot be compiled by clang), but did not occur in practice.
    objc_defines += objc.define

  output_lib = ctx.new_file(module_name + ".a")
  output_module = ctx.new_file(module_name + ".swiftmodule")
  output_header = ctx.new_file(ctx.label.name + "-Swift.h")
  output_file_map = ctx.new_file(ctx.label.name + ".output_file_map.json")

  output_map = struct()
  output_objs = []
  for source in ctx.files.srcs:
    obj = ctx.new_file(source.basename + ".o")
    output_objs.append(obj)

    output_map += struct(**{source.path: struct(object=obj.path)})

  # Write down the output file map for this compilation, to be used with
  # -output-file-map flag.
  # It's a JSON file that maps each source input (.swift) to its outputs
  # (.o, .bc, .d, ...)
  # Example:
  #   {'foo.swift':
  #       {'object': 'foo.o', 'bitcode': 'foo.bc', 'dependencies': 'foo.d'}}
  # There's currently no documentation on this option, however all of the keys
  # are listed here https://github.com/apple/swift/blob/swift-2.2.1-RELEASE/include/swift/Driver/Types.def
  ctx.file_action(output=output_file_map, content=output_map.to_json())

  srcs_args = [f.path for f in ctx.files.srcs]

  # Include each swift module's parent directory for imports to work.
  include_dirs = set([x.dirname for x in dep_modules])

  include_args = ["-I%s" % d for d in include_dirs + objc_includes]
  framework_args = ["-F%s" % x for x in framework_dirs]

  clang_args = _intersperse(
      "-Xcc",

      # Add the current directory to clang's search path.
      # This instance of clang is spawned by swiftc to compile module maps and
      # is not passed the current directory as a search path by default.
      ["-iquote", "."]

      # Pass DEFINE or copt values from objc configuration and rules to clang
      + ["-D" + x for x in objc_defines] + ctx.fragments.objc.copts
      + _clang_compilation_mode_flags(ctx)

      # Load module maps explicitly instead of letting Clang discover them on
      # search paths. This is needed to avoid a case where Clang may load the
      # same header both in modular and non-modular contexts, leading to
      # duplicate definitions in the same file.
      # https://llvm.org/bugs/show_bug.cgi?id=19501
      + ["-fmodule-map-file=%s" % x.path for x in objc_module_maps])

  args = [
      "swiftc",
      "-emit-object",
      "-emit-module-path",
      output_module.path,
      "-module-name",
      module_name,
      "-emit-objc-header-path",
      output_header.path,
      "-parse-as-library",
      "-target",
      target,
      "-sdk",
      apple_toolchain.sdk_dir(),
      "-module-cache-path",
      module_cache_path(ctx),
      "-output-file-map",
      output_file_map.path,
  ] + _swift_compilation_mode_flags(ctx)

  args.extend(srcs_args)
  args.extend(include_args)
  args.extend(framework_args)
  args.extend(clang_args)

  xcrun_action(ctx,
               inputs=ctx.files.srcs + dep_modules + list(objc_files) +
               [output_file_map],
               outputs=[output_module, output_header] + output_objs,
               mnemonic="SwiftCompile",
               arguments=args,
               use_default_shell_env=False,
               progress_message=("Compiling Swift module %s (%d files)" %
                                 (ctx.label.name, len(ctx.files.srcs))))

  xcrun_action(ctx,
               inputs=output_objs,
               outputs=(output_lib,),
               mnemonic="SwiftArchive",
               arguments=[
                   "libtool", "-static", "-o", output_lib.path
               ] + [x.path for x in output_objs],
               progress_message=("Archiving Swift objects %s" % ctx.label.name))

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
        "srcs": attr.label_list(allow_files = [".swift"]),
        "deps": attr.label_list(providers=[["swift"], ["objc"]]),
        "module_name": attr.string(mandatory=False),
        "_xcrunwrapper": attr.label(
            executable=True,
            default=Label(XCRUNWRAPPER_LABEL))},
    fragments = ["apple", "objc"],
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
