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

load("shared",
     "xcrun_action",
     "XCRUNWRAPPER_LABEL",
     "module_cache_path",
     "label_scoped_path")

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

def _swift_target(cpu, platform, sdk_version):
  """Returns a target triplet for Swift compiler."""
  platform_string = str(platform.platform_type)
  if platform_string not in ["ios", "watchos"]:
    fail("Platform '%s' is not supported" % platform_string)

  return "%s-apple-%s%s" % (cpu, platform_string, sdk_version)

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

def _swift_bitcode_flags(ctx):
  """Returns bitcode flags based on selected mode."""
  # TODO(dmishe): Remove this when bitcode_mode is available by default.
  if hasattr(ctx.fragments.apple, "bitcode_mode"):
    mode = str(ctx.fragments.apple.bitcode_mode)
    if mode == "embedded":
      return ["-embed-bitcode"]
    elif mode == "embedded_markers":
      return ["-embed-bitcode-marker"]

  return []

def _module_name(ctx):
  """Returns a module name for the given rule context."""
  return ctx.label.package.lstrip("//").replace("/", "_") + "_" + ctx.label.name

def _swift_library_impl(ctx):
  """Implementation for swift_library Skylark rule."""
  # TODO(b/29772303): Assert xcode version.
  apple_fragment = ctx.fragments.apple

  cpu = apple_fragment.single_arch_cpu
  platform = apple_fragment.single_arch_platform

  # TODO(cparsons): Remove after blaze release.
  if hasattr(ctx.fragments.apple, "minimum_os_for_platform_type"):
    target_os = ctx.fragments.apple.minimum_os_for_platform_type(
        apple_common.platform_type.ios)
  else:
    target_os = ctx.fragments.objc.ios_minimum_os
  target = _swift_target(cpu, platform, target_os)
  apple_toolchain = apple_common.apple_toolchain()

  module_name = ctx.attr.module_name or _module_name(ctx)

  # A list of paths to pass with -F flag.
  framework_dirs = set([
      apple_toolchain.platform_developer_framework_dir(apple_fragment)])

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

  # A unique path for rule's outputs.
  objs_outputs_path = label_scoped_path(ctx, "_objs/")

  output_lib = ctx.new_file(objs_outputs_path + module_name + ".a")
  output_module = ctx.new_file(objs_outputs_path + module_name + ".swiftmodule")

  # These filenames are guaranteed to be unique, no need to scope.
  output_header = ctx.new_file(ctx.label.name + "-Swift.h")
  swiftc_output_map_file = ctx.new_file(ctx.label.name + ".output_file_map.json")

  swiftc_output_map = struct()  # Maps output types to paths.
  output_objs = []  # Object file outputs, used in archive action.
  swiftc_outputs = []  # Other swiftc outputs that aren't processed further.
  for source in ctx.files.srcs:
    basename = source.basename
    obj = ctx.new_file(objs_outputs_path + basename + ".o")
    partial_module = ctx.new_file(objs_outputs_path + basename +
                                  ".partial_swiftmodule")
    output_objs.append(obj)
    swiftc_outputs.append(partial_module)

    swiftc_output_map += struct(**{
        source.path: struct(object=obj.path, swiftmodule=partial_module.path)})

  # Write down the intermediate outputs map for this compilation, to be used
  # with -output-file-map flag.
  # It's a JSON file that maps each source input (.swift) to its outputs
  # (.o, .bc, .d, ...)
  # Example:
  #   {'foo.swift':
  #       {'object': 'foo.o', 'bitcode': 'foo.bc', 'dependencies': 'foo.d'}}
  # There's currently no documentation on this option, however all of the keys
  # are listed here https://github.com/apple/swift/blob/swift-2.2.1-RELEASE/include/swift/Driver/Types.def
  ctx.file_action(output=swiftc_output_map_file, content=swiftc_output_map.to_json())

  srcs_args = [f.path for f in ctx.files.srcs]

  # Include each swift module's parent directory for imports to work.
  include_dirs = set([x.dirname for x in dep_modules])

  # Include the parent directory of the resulting module so LLDB can find it.
  include_dirs += set([output_module.dirname])

  include_args = ["-I%s" % d for d in include_dirs + objc_includes]
  framework_args = ["-F%s" % x for x in framework_dirs]
  define_args = ["-D%s" % x for x in ctx.attr.defines]

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
      swiftc_output_map_file.path,
  ] + _swift_compilation_mode_flags(ctx) + _swift_bitcode_flags(ctx)

  args.extend(srcs_args)
  args.extend(include_args)
  args.extend(framework_args)
  args.extend(clang_args)
  args.extend(define_args)
  args.extend(ctx.attr.copts)

  xcrun_action(
      ctx,
      inputs=ctx.files.srcs + dep_modules + list(objc_files) +
      [swiftc_output_map_file],
      outputs=[output_module, output_header] + output_objs + swiftc_outputs,
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
        "defines": attr.string_list(mandatory=False, allow_empty=True),
        "copts": attr.string_list(mandatory=False, allow_empty=True),
        "_xcrunwrapper": attr.label(
            executable=True,
            cfg="host",
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
      the module name is the target path with all special symbols replaced
      by "_", e.g. //foo:bar can be imported as "foo_bar".
  copts: A list of flags passed to swiftc command line.
  defines: A list of values for build configuration options (-D). These values
      can be then used for conditional compilation blocks in code. For example:

      BUILD:
        swift_library(
          defines = ["VALUE"]
        )

      Code:
        #if VALUE
        #endif
"""
