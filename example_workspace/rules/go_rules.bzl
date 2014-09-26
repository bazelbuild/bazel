# Copyright 2014 Google Inc. All rights reserved.
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

"""These are bare-bones Go rules.

Several issues:

- For "a/b/c.go", the go tool creates library "a/b.a" with import path
"a/b".  We can probably simulate this with symlink trees.

- Dependencies are not enforced; a symlink tree might help here too.

- Hardcoded to 6g from the GC suite. We should be able to support GCC
  and derive 6g from the CPU (from which configuration?)

- It would be nice to be able to create a full-fledged Go
  configuration in Skylark.

- It would be nice to support zero-configuration
go_library()/go_binary()/go_test() rules:

  * name defaults to basename of directory
  * srcs defaults to *.go

- does not support checked in compilers.

- No C++ interop.

- deps must be populated by hand or using Glaze.

- go_test must have both test and non-test files in srcs.
"""

go_filetype = filetype([".go"])
go_lib_filetype = filetype([".a"])

def go_compile_cmd(ctx, sources, out_lib):
  cmd = ctx.executable.go_tool.path + " tool 6g "
  cmd += " -o " + out_lib.path  + " "
  cmd += " -pack"

  # Import path.
  cmd += " -I " + str(ctx.configuration.bin_dir)[:-9]

  # Set -p to the import path of the library, ie.
  # (ctx.label.package + "/" ctx.label.name)  for now.
  cmd += " " + files.join_exec_paths(" ", sources)
  return cmd

def go_library_impl(ctx):
  sources = ctx.files.srcs
  out_lib = ctx.outputs.lib
  deps = ctx.files.deps

  ctx.action(
      inputs = sources + deps,
      outputs = [out_lib],
      mnemonic = "GoCompile",
      command = go_compile_cmd(ctx, sources, out_lib))

  out_nset = nset("STABLE_ORDER", [out_lib])
  return struct(
    files_to_build= out_nset,
    go_library_object = out_nset)


def go_link_action(ctx, lib, executable):
  cmd = ctx.file("go_tool").path + " tool 6l "

  # Link search path.
  cmd += " -L " + str(ctx.configuration.bin_dir)[:-9]
  cmd += " -o " + executable.path
  cmd += " " + lib.path
  ctx.action(
      inputs = [lib],
      outputs = [executable],
      mnemonic = "GoLink",
      command = cmd)


def go_binary_impl(ctx):
  lib_result = go_library_impl(ctx)
  executable = ctx.outputs.executable
  lib_out = ctx.outputs.lib

  go_link_action(ctx, lib_out, executable)
  return struct(
      files_to_build = (
          nset("STABLE_ORDER", [executable]) + lib_result.files_to_build))


def go_test_impl(ctx):
  lib_result = go_library_impl(ctx)
  main_go = ctx.outputs.main_go

  go_import = ctx.label.package + "/"  + ctx.label.name

  generator = ctx.executable("test_generator")
  main_cmd = generator.path

  # Would be nice to use transitive info provider to get at sources of
  # a dependent library.
  sources = ctx.files.srcs
  main_cmd += " --package " + go_import
  main_cmd += " --output " + ctx.outputs.main_go.path
  main_cmd += " " + files.join_exec_paths(" ", sources)

  ctx.action(
      # confusing message if you do [sources] + (something else).
      inputs = sources + [generator],
      outputs = [main_go],
      mnemonic = "GoTestGenTest",
      command = main_cmd)

  ctx.action(
      inputs = [main_go, ctx.outputs.lib],
      outputs = [ctx.outputs.main_lib],
      command = go_compile_cmd(ctx, [main_go], ctx.outputs.main_lib),
      mnemonic = "GoCompileTest")

  go_link_action(ctx, ctx.outputs.main_lib,  ctx.outputs.executable)

  runfiles = ctx.runfiles(collect_data = True, files = [ctx.outputs.executable])
  return struct(
     runfiles=runfiles,
     )

go_library_attrs = {
    "data":  attr.label_list(
        file_types=ANY_FILE, rule_classes=NO_RULE,
        cfg=DATA_CFG),
    "srcs": attr.label_list(
        file_types=go_filetype),
    "deps": attr.label_list(
        file_types=NO_FILE,
        providers=["go_library_object"]),
    "go_tool": attr.label(
        default=label("//tools/go:go"),
        file_types=ANY_FILE,
        cfg=HOST_CFG, executable=True),
    }

go_library_outputs = {
    "lib": "%{name}.a",
    }

go_library = rule(
    go_library_impl,
    attr = go_library_attrs,
    outputs =go_library_outputs)

go_binary = rule(
    go_binary_impl,
    executable = True,
    attr = go_library_attrs,
    outputs = go_library_outputs)

go_test = rule(
    go_test_impl,
    executable = True,
    test = True,
    attr = go_library_attrs + {
      "test_generator": attr.label(
          default=label("//tools/go:generate_test_main"),
          cfg=HOST_CFG, flags=["EXECUTABLE"])
      },
    outputs =  {
      "lib" : "%{name}.a",
      "main_lib": "%{name}_main_test.a",
      "main_go": "%{name}_main_test.go",
        })
