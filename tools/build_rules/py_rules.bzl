# Copyright 2014 The Bazel Authors. All rights reserved.
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

ZIP_PATH = "/usr/bin/zip"

py_file_types = FileType([".py"])


def collect_transitive_sources(ctx):
  source_files = set(order="compile")
  for dep in ctx.attr.deps:
    source_files += dep.transitive_py_files

  source_files += py_file_types.filter(ctx.files.srcs)
  return source_files


def py_library_impl(ctx):
  transitive_sources = collect_transitive_sources(ctx)
  return struct(
      files = set(),
      transitive_py_files = transitive_sources)


def py_binary_impl(ctx):
  main_file = py_file_types.filter(ctx.files.srcs)[0]
  transitive_sources = collect_transitive_sources(ctx)
  deploy_zip = ctx.outputs.deploy_zip

  deploy_zip_nomain = ctx.new_file(
      ctx.configuration.bin_dir, deploy_zip, ".nomain.zip")

  # This is not very scalable, because we just construct a huge string instead
  # of using a nested set. We need to do it this way because Skylark currently
  # does not support actions with non-artifact executables but with an
  # argument list (instead of just a single command)
  command = ZIP_PATH +" -q " + deploy_zip_nomain.path + " " + " ".join([f.path for f in transitive_sources])
  ctx.action(
      inputs = list(transitive_sources),
      outputs = [ deploy_zip_nomain ],
      mnemonic = "PyZip",
      command = command,
      use_default_shell_env = False)

  dirs = [f.path[:f.path.rfind('/')] for f in transitive_sources]
  outdir = deploy_zip.path + ".out"

  # Add __init__.py files and the __main__.py driver.
  main_cmd = ("mkdir -p %s && " % outdir +
              " cp %s %s/__main__.py && " % (main_file.path, outdir) +
              " cp %s %s/main.zip && " % (deploy_zip_nomain.path, outdir) +
              " (cd %s && " % outdir +
              "  mkdir -p %s && " % " ".join(dirs) +
              "  find . -type d -exec touch -t 198001010000 '{}'/__init__.py ';' && " +
              "  chmod +w main.zip && " +
              "  %s -quR main.zip $(find . -type f ) ) && " % (ZIP_PATH) +
              " mv %s/main.zip %s " % (outdir, deploy_zip.path))

  ctx.action(
      inputs = [ deploy_zip_nomain, main_file ],
      outputs = [ deploy_zip ],
      mnemonic = "PyZipMain",
      command = main_cmd)

  executable = ctx.outputs.executable
  ctx.action(
      inputs = [ deploy_zip, ],
      outputs = [ executable, ],
      command = "echo '#!/usr/bin/env python' | cat - %s > %s && chmod +x %s" % (
          deploy_zip.path, executable.path, executable.path))

  runfiles_files = transitive_sources + [executable]

  runfiles = ctx.runfiles(transitive_files = runfiles_files,
                          collect_default = True)

  files_to_build = set([deploy_zip, executable])
  return struct(files = files_to_build, runfiles = runfiles)


py_srcs_attr = attr.label_list(allow_files = py_file_types)

py_deps_attr = attr.label_list(
    providers = ["transitive_py_files"],
    allow_files = False)

py_attrs = {
    "srcs": py_srcs_attr,
    "deps": py_deps_attr }

py_library = rule(
    py_library_impl,
    attrs = py_attrs)

py_binary_outputs = {
    "deploy_zip": "%{name}.zip"
    }

py_binary = rule(
    py_binary_impl,
    executable = True,
    attrs = py_attrs,
    outputs = py_binary_outputs)

py_test = rule(
  py_binary_impl,
  executable = True,
  attrs = py_attrs,
  outputs = py_binary_outputs)
