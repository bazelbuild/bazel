# Copyright 2015 The Bazel Authors. All rights reserved.
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

"""Rust rules for Bazel"""

RUST_FILETYPE = FileType([".rs"])
A_FILETYPE = FileType([".a"])

# Used by rust_docs
HTML_MD_FILETYPE = FileType([".html", ".md"])
CSS_FILETYPE = FileType([".css"])

ZIP_PATH = "/usr/bin/zip"

def _relative(src_path, dest_path):
  """Returns the relative path from src_path to dest_path."""
  src_parts = src_path.split("/")
  dest_parts = dest_path.split("/")
  n = 0
  done = False
  for src_part, dest_part in zip(src_parts, dest_parts):
    if src_part != dest_part:
      break
    n += 1

  relative_path = ""
  for i in range(n, len(src_parts)):
    relative_path += "../"
  relative_path += "/".join(dest_parts[n:])

  return relative_path

def _create_setup_cmd(lib, deps_dir):
  """
  Helper function to construct a command for symlinking a library into the
  deps directory.
  """
  return (
      "ln -sf " + _relative(deps_dir, lib.path) + " " +
      deps_dir + "/" + lib.basename + "\n"
  )

def _setup_deps(deps, name, working_dir, is_library=False):
  """
  Walks through dependencies and constructs the necessary commands for linking
  to all the necessary dependencies.

  Args:
    deps: List of Labels containing deps from ctx.attr.deps.
    name: Name of the current target.
    working_dir: The output directory for the current target's outputs.
    is_library: True the current target is a rust_library target, False
        otherwise.

  Returns:
    Returns a struct containing the following fields:
      libs:
      transitive_libs:
      setup_cmd:
      search_flags:
      link_flags:
  """
  deps_dir = working_dir + "/" + name + ".deps"
  setup_cmd = ["rm -rf " + deps_dir + "; mkdir " + deps_dir + "\n"]

  has_rlib = False
  has_native = False

  libs = set()
  transitive_libs = set()
  symlinked_libs = set()
  link_flags = []
  for dep in deps:
    if hasattr(dep, "rust_lib"):
      # This dependency is a rust_library
      libs += [dep.rust_lib]
      transitive_libs += [dep.rust_lib] + dep.transitive_libs
      symlinked_libs += [dep.rust_lib] + dep.transitive_libs
      link_flags += [(
          "--extern " + dep.label.name + "=" +
          deps_dir + "/" + dep.rust_lib.basename
      )]
      has_rlib = True

    elif hasattr(dep, "cc"):
      if not is_library:
        fail("Only rust_library targets can depend on cc_library")

      # This dependency is a cc_library
      native_libs = A_FILETYPE.filter(dep.cc.libs)
      libs += native_libs
      transitive_libs += native_libs
      symlinked_libs += native_libs
      link_flags += ["-l static=" + dep.label.name]
      has_native = True

    else:
      fail(("rust_library" if is_library else "rust_binary and rust_test") +
           " targets can only depend on rust_library " +
           ("or cc_library " if is_library else "") + "targets")

  for symlinked_lib in symlinked_libs:
    setup_cmd += [_create_setup_cmd(symlinked_lib, deps_dir)]

  search_flags = []
  if has_rlib:
    search_flags += ["-L dependency=%s" % deps_dir]
  if has_native:
    search_flags += ["-L native=%s" % deps_dir]

  return struct(
      libs = list(libs),
      transitive_libs = list(transitive_libs),
      setup_cmd = setup_cmd,
      search_flags = search_flags,
      link_flags = link_flags)

def _get_features_flags(features):
  """
  Constructs a string containing the feature flags from the features specified
  in the features attribute.
  """
  features_flags = []
  for feature in features:
    features_flags += ["--cfg feature=\\\"%s\\\"" % feature]
  return features_flags

def _rust_toolchain(ctx):
  return struct(
      rustc_path = ctx.file._rustc.path,
      rustc_lib_path = ctx.files._rustc_lib[0].dirname,
      rustlib_path = ctx.files._rustlib[0].dirname,
      rustdoc_path = ctx.file._rustdoc.path)

def _build_rustc_command(ctx, crate_type, src, output_dir, depinfo,
                         extra_flags=[]):
  """Builds the rustc command.

  Constructs the rustc command used to build the current target.

  Args:
    ctx: The ctx object for the current target.
    crate_type: The type of crate to build ("lib" or "bin")
    src: The path to the crate root source file ("lib.rs" or "main.rs")
    output_dir: The output directory for the target.
    depinfo: Struct containing information about dependencies as returned by
        _setup_deps
    extra_flags: Additional command line flags.

  Return:
    String containing the rustc command.
  """

  # Paths to the Rust compiler and standard libraries.
  toolchain = _rust_toolchain(ctx)

  # Paths to cc (for linker) and ar
  cpp_fragment = ctx.fragments.cpp
  cc = cpp_fragment.compiler_executable
  ar = cpp_fragment.ar_executable
  # Currently, the CROSSTOOL config for darwin sets ar to "libtool". Because
  # rust uses ar-specific flags, use /usr/bin/ar in this case.
  # TODO(dzc): This is not ideal. Remove this workaround once ar_executable
  # always points to an ar binary.
  ar_str = "%s" % ar
  if ar_str.find("libtool", 0) != -1:
    ar = "/usr/bin/ar"

  # Construct features flags
  features_flags = _get_features_flags(ctx.attr.crate_features)

  return " ".join([
      "set -e;",
      " ".join(depinfo.setup_cmd),
      "LD_LIBRARY_PATH=%s" % toolchain.rustc_lib_path,
      "DYLD_LIBRARY_PATH=%s" % toolchain.rustc_lib_path,
      toolchain.rustc_path,
      src,
      "--crate-name %s" % ctx.label.name,
      "--crate-type %s" % crate_type,
      "-C opt-level=3",
      "--codegen ar=%s" % ar,
      "--codegen linker=%s" % cc,
      "-L all=%s" % toolchain.rustlib_path,
      " ".join(extra_flags),
      " ".join(features_flags),
      "--out-dir %s" % output_dir,
      "--emit=dep-info,link",
      " ".join(depinfo.search_flags),
      " ".join(depinfo.link_flags),
      " ".join(ctx.attr.rustc_flags),
  ])

def _find_crate_root_src(srcs, file_names=["lib.rs"]):
  """Finds the source file for the crate root."""
  for src in srcs:
    if src.basename in file_names:
      return src.path
  fail("No %s source file found." % " or ".join(file_names), "srcs")

def _rust_library_impl(ctx):
  """
  Implementation for rust_library Skylark rule.
  """

  # Find lib.rs
  lib_rs = _find_crate_root_src(ctx.files.srcs)

  # Output library
  rust_lib = ctx.outputs.rust_lib
  output_dir = rust_lib.dirname

  # Dependencies
  depinfo = _setup_deps(ctx.attr.deps,
                        ctx.label.name,
                        output_dir,
                        is_library=True)

  # Build rustc command
  cmd = _build_rustc_command(
      ctx = ctx,
      crate_type = "lib",
      src = lib_rs,
      output_dir = output_dir,
      depinfo = depinfo)

  # Compile action.
  compile_inputs = (
      ctx.files.srcs +
      ctx.files.data +
      depinfo.libs +
      [ctx.file._rustc] +
      ctx.files._rustc_lib +
      ctx.files._rustlib)

  ctx.action(
      inputs = compile_inputs,
      outputs = [rust_lib],
      mnemonic = 'Rustc',
      command = cmd,
      use_default_shell_env = True,
      progress_message = ("Compiling Rust library %s (%d files)"
                          % (ctx.label.name, len(ctx.files.srcs))))

  return struct(
      files = set([rust_lib]),
      rust_srcs = ctx.files.srcs,
      rust_deps = ctx.attr.deps,
      transitive_libs = depinfo.transitive_libs,
      rust_lib = rust_lib)

def _rust_binary_impl_common(ctx, extra_flags = []):
  """Implementation for rust_binary Skylark rule."""

  # Find main.rs.
  main_rs = _find_crate_root_src(ctx.files.srcs, ["main.rs"])

  # Output binary
  rust_binary = ctx.outputs.executable
  output_dir = rust_binary.dirname

  # Dependencies
  depinfo = _setup_deps(ctx.attr.deps,
                        ctx.label.name,
                        output_dir,
                        is_library=False)

  # Build rustc command.
  cmd = _build_rustc_command(ctx = ctx,
                             crate_type = "bin",
                             src = main_rs,
                             output_dir = output_dir,
                             depinfo = depinfo,
                             extra_flags = extra_flags)

  # Compile action.
  compile_inputs = (
      ctx.files.srcs +
      ctx.files.data +
      depinfo.libs +
      [ctx.file._rustc] +
      ctx.files._rustc_lib +
      ctx.files._rustlib)

  ctx.action(
      inputs = compile_inputs,
      outputs = [rust_binary],
      mnemonic = 'Rustc',
      command = cmd,
      use_default_shell_env = True,
      progress_message = ("Compiling Rust binary %s (%d files)"
                          % (ctx.label.name, len(ctx.files.srcs))))

  return struct(rust_srcs = ctx.files.srcs,
                rust_deps = ctx.attr.deps)

def _rust_binary_impl(ctx):
  """
  Implementation for rust_binary Skylark rule.
  """
  return _rust_binary_impl_common(ctx)

def _rust_test_impl(ctx):
  """
  Implementation for rust_test and rust_bench_test Skylark rules.
  """
  return _rust_binary_impl_common(ctx, ["--test"])

def _build_rustdoc_flags(ctx):
  """Collects the rustdoc flags."""
  doc_flags = []
  doc_flags += [
      "--markdown-css %s" % css.path for css in ctx.files.markdown_css]
  if hasattr(ctx.file, "html_in_header"):
    doc_flags += ["--html-in-header %s" % ctx.file.html_in_header.path]
  if hasattr(ctx.file, "html_before_content"):
    doc_flags += ["--html-before-content %s" %
                  ctx.file.html_before_content.path]
  if hasattr(ctx.file, "html_after_content"):
    doc_flags += ["--html-after-content %s"]
  return doc_flags

def _rust_docs_impl(ctx):
  """Implementation of the rust_docs rule."""
  rust_doc_zip = ctx.outputs.rust_doc_zip

  # Gather attributes about the rust_library target to generated rustdocs for.
  target = struct(name = ctx.attr.dep.label.name,
                  srcs = ctx.attr.dep.rust_srcs,
                  deps = ctx.attr.dep.rust_deps)

  # Find lib.rs
  lib_rs = _find_crate_root_src(target.srcs, ["lib.rs", "main.rs"])

  # Dependencies
  output_dir = rust_doc_zip.dirname
  depinfo = _setup_deps(target.deps,
                        target.name,
                        output_dir,
                        is_library=False)

  # Rustdoc flags.
  doc_flags = _build_rustdoc_flags(ctx)

  # Build rustdoc command.
  toolchain = _rust_toolchain(ctx)
  docs_dir = rust_doc_zip.dirname + "/_rust_docs"
  doc_cmd = " ".join(
      ["set -e"] +
      depinfo.setup_cmd + [
          "rm -rf %s;" % docs_dir,
          "mkdir %s;" % docs_dir,
          "LD_LIBRARY_PATH=%s" % toolchain.rustc_lib_path,
          "DYLD_LIBRARY_PATH=%s" % toolchain.rustc_lib_path,
          toolchain.rustdoc_path,
          lib_rs,
          "--crate-name %s" % target.name,
          "-L all=%s" % toolchain.rustlib_path,
          "-o %s" % docs_dir,
      ] +
      doc_flags +
      depinfo.search_flags +
      depinfo.link_flags + [
          "&&",
          "(cd %s" % docs_dir,
          "&&",
          ZIP_PATH,
          "-qR",
          rust_doc_zip.basename,
          "$(find . -type f) )",
          "&&",
          "mv %s/%s %s" % (docs_dir, rust_doc_zip.basename, rust_doc_zip.path),
      ])

  # Rustdoc action
  rustdoc_inputs = (target.srcs +
                    depinfo.libs +
                    [ctx.file._rustdoc] +
                    ctx.files._rustc_lib +
                    ctx.files._rustlib)

  ctx.action(
      inputs = rustdoc_inputs,
      outputs = [rust_doc_zip],
      mnemonic = 'Rustdoc',
      command = doc_cmd,
      use_default_shell_env = True,
      progress_message = ("Generating rustdoc for %s (%d files)"
                          % (target.name, len(target.srcs))))

_rust_common_attrs = {
    "srcs": attr.label_list(allow_files = RUST_FILETYPE),
    "data": attr.label_list(allow_files = True, cfg = DATA_CFG),
    "deps": attr.label_list(),
    "crate_features": attr.string_list(),
    "rustc_flags": attr.string_list(),
}

_rust_toolchain_attrs = {
    "_rustc": attr.label(
        default = Label("//tools/build_rules/rust:rustc"),
        executable = True,
        single_file = True),
    "_rustc_lib": attr.label(
        default = Label("//tools/build_rules/rust:rustc_lib")),
    "_rustlib": attr.label(default = Label("//tools/build_rules/rust:rustlib")),
    "_rustdoc": attr.label(
        default = Label("//tools/build_rules/rust:rustdoc"),
        executable = True,
        single_file = True),
}

rust_library = rule(
    _rust_library_impl,
    attrs = _rust_common_attrs + _rust_toolchain_attrs,
    outputs = {
        "rust_lib": "lib%{name}.rlib",
    },
    fragments = ["cpp"],
)

rust_binary = rule(
    _rust_binary_impl,
    executable = True,
    attrs = _rust_common_attrs + _rust_toolchain_attrs,
    fragments = ["cpp"],
)

rust_test = rule(
    _rust_test_impl,
    executable = True,
    attrs = _rust_common_attrs + _rust_toolchain_attrs,
    test = True,
    fragments = ["cpp"],
)

rust_bench_test = rule(
    _rust_test_impl,
    executable = True,
    attrs = _rust_common_attrs + _rust_toolchain_attrs,
    test = True,
    fragments = ["cpp"],
)

_rust_doc_attrs = {
    "dep": attr.label(mandatory = True),
    "markdown_css": attr.label_list(allow_files = CSS_FILETYPE),
    "html_in_header": attr.label(allow_files = HTML_MD_FILETYPE),
    "html_before_content": attr.label(allow_files = HTML_MD_FILETYPE),
    "html_after_content": attr.label(allow_files = HTML_MD_FILETYPE),
}

rust_docs = rule(
    _rust_docs_impl,
    attrs = _rust_doc_attrs + _rust_toolchain_attrs,
    outputs = {
        "rust_doc_zip": "%{name}-docs.zip",
    },
)
