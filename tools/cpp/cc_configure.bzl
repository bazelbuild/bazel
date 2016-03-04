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
"""Rules for configuring the C++ toolchain (experimental)."""


def _get_value(it):
  """Convert `it` in serialized protobuf format."""
  if type(it) == "int":
    return str(it)
  elif type(it) == "bool":
    return "true" if it else "false"
  else:
    return "\"%s\"" % it


def _build_crosstool(d, prefix="  "):
  """Convert `d` to a string version of a CROSSTOOL file content."""
  lines = []
  for k in d:
    if type(d[k]) == "list":
      for it in d[k]:
        lines.append("%s%s: %s" % (prefix, k, _get_value(it)))
    else:
      lines.append("%s%s: %s" % (prefix, k, _get_value(d[k])))
  return "\n".join(lines)


def _build_tool_path(d):
  """Build the list of tool_path for the CROSSTOOL file."""
  lines = []
  for k in d:
    lines.append("  tool_path {name: \"%s\" path: \"%s\" }" % (k, d[k]))
  return "\n".join(lines)


def _which(ctx, cmd, default):
  """A wrapper around ctx.which() to provide a fallback value."""
  result = ctx.which(cmd)
  return default if result == None else str(result)


def _get_tool_paths(ctx, darwin, cc):
  """Compute the path to the various tools."""
  return {k: _which(ctx, k, "/usr/bin/" + k)
          for k in [
              "ld",
              "cpp",
              "dwp",
              "gcov",
              "nm",
              "objcopy",
              "objdump",
              "strip",
          ]} + {
              "gcc": cc,
              "ar": "/usr/bin/libtool"
                    if darwin else _which(ctx, "ar", "/usr/bin/ar")
          }


def _ld_library_paths(ctx):
  """Use ${LD_LIBRARY_PATH} to compute the list -Wl,rpath flags."""
  if "LD_LIBRARY_PATH" in ctx.os.environ:
    result = []
    for p in ctx.os.environ["LD_LIBRARY_PATH"].split(":"):
      p = ctx.path(p)  # Normalize the path
      result.append("-Wl,rpath," + p)
      result.append("-L" + p)
    return result
  else:
    return []


def _get_cpu_value(ctx):
  """Compute the cpu_value based on the OS name."""
  return "darwin" if ctx.os.name.lower().startswith("mac os") else "k8"


_INC_DIR_MARKER_BEGIN = "#include <...> search starts here:"
_INC_DIR_MARKER_END = "End of search list."


def _get_cxx_inc_directories(ctx, cc):
  """Compute the list of default C++ include directories."""
  result = ctx.execute([cc, "-E", "-xc++", "-", "-v"])
  index1 = result.stderr.find(_INC_DIR_MARKER_BEGIN)
  if index1 == -1:
    return []
  index2 = result.stderr.find(_INC_DIR_MARKER_END, index1)
  if index2 == -1:
    return []
  inc_dirs = result.stderr[index1 + len(_INC_DIR_MARKER_BEGIN):index2].strip()
  return [ctx.path(p.strip()) for p in inc_dirs.split("\n")]


def _add_option_if_supported(ctx, cc, option):
  """Checks that `option` is supported by the C compiler."""
  result = ctx.execute([cc, option])
  return [option] if result.stderr.find(option) == -1 else []


def _crosstool_content(ctx, cc, cpu_value, darwin):
  """Return the content for the CROSSTOOL file, in a dictionary."""
  return {
      "abi_version": "local",
      "abi_libc_version": "local",
      "builtin_sysroot": "",
      "compiler": "compiler",
      "host_system_name": "local",
      "needsPic": True,
      "supports_gold_linker": False,
      "supports_incremental_linker": False,
      "supports_fission": False,
      "supports_interface_shared_objects": False,
      "supports_normalizing_ar": False,
      "supports_start_end_lib": False,
      "supports_thin_archives": False,
      "target_libc": "macosx" if darwin else "local",
      "target_cpu": cpu_value,
      "target_system_name": "local",
      "cxx_flag": "-std=c++0x",
      "linker_flag": [
          "-lstdc++",
          # Anticipated future default.
          "-no-canonical-prefixes"
      ] + (["-undefined", "dynamic_lookup"] if darwin else [
          "-B/usr/bin",
          # Have gcc return the exit code from ld.
          "-pass-exit-codes",
          # Stamp the binary with a unique identifier.
          "-Wl,--build-id=md5",
          "-Wl,--hash-style=gnu"
          # Gold linker only? Can we enable this by default?
          # "-Wl,--warn-execstack",
          # "-Wl,--detect-odr-violations"
      ]) + _ld_library_paths(ctx),
      "ar_flag": ["-static", "-s", "-o"] if darwin else [],
      "cxx_builtin_include_directory": _get_cxx_inc_directories(ctx, cc),
      "objcopy_embed_flag": ["-I", "binary"],
      "unfiltered_cxx_flag": [
          # Anticipated future default.
          "-no-canonical-prefixes",
      ] + ([] if darwin else ["-fno-canonical-system-headers"]) + [
          # Make C++ compilation deterministic. Use linkstamping instead of these
          # compiler symbols.
          "-Wno-builtin-macro-redefined",
          "-D__DATE__=\\\"redacted\\\"",
          "-D__TIMESTAMP__=\\\"redacted\\\"",
          "-D__TIME__=\\\"redacted\\\""
      ],
      "compiler_flag": [
          # Security hardening on by default.
          # Conservative choice; -D_FORTIFY_SOURCE=2 may be unsafe in some cases.
          # We need to undef it before redefining it as some distributions now have
          # it enabled by default.
          "-U_FORTIFY_SOURCE",
          "-D_FORTIFY_SOURCE=1",
          "-fstack-protector",
          # All warnings are enabled. Maybe enable -Werror as well?
          "-Wall",
          # Enable a few more warnings that aren't part of -Wall.
      ] + (["-Wthread-safety", "-Wself-assign"] if darwin else [
          "-Wunused-but-set-parameter",
          # Disable some that are problematic.
          "-Wno-free-nonheap-object",  # has false positives
          "-Wl,-z,-relro,-z,now"
      ]) + (
          # Enable coloring even if there's no attached terminal. Bazel removes the
          # escape sequences if --nocolor is specified.
          _add_option_if_supported(ctx, cc, "-fcolor-diagnostics")) + [
              # Keep stack frames for debugging, even in opt mode.
              "-fno-omit-frame-pointer",
          ],
  }


def _opt_content(darwin):
  """Return the content of the opt specific section of the CROSSTOOL file."""
  return {
      "compiler_flag": [
          # No debug symbols.
          # Maybe we should enable https://gcc.gnu.org/wiki/DebugFission for opt or
          # even generally? However, that can't happen here, as it requires special
          # handling in Bazel.
          "-g0",

          # Conservative choice for -O
          # -O3 can increase binary size and even slow down the resulting binaries.
          # Profile first and / or use FDO if you need better performance than this.
          "-O2",

          # Disable assertions
          "-DNDEBUG",

          # Removal of unused code and data at link time (can this increase binary size in some cases?).
          "-ffunction-sections",
          "-fdata-sections"
      ],
      "linker_flag": [] if darwin else ["-Wl,--gc-sections"]
  }


def _dbg_content():
  """Return the content of the dbg specific section of the CROSSTOOL file."""
  # Enable debug symbols
  return {"compiler_flag": "-g"}


def _find_cc(ctx):
  """Find the C++ compiler."""
  if "CC" in ctx.os.environ:
    return ctx.path(ctx.os.environ["CC"])
  else:
    cc = ctx.which("gcc")
    if cc == None:
      fail(
          "Cannot find gcc, either correct your path or set the CC" +
          " ennvironment variable")
    return cc


def _tpl(ctx, tpl, substitutions={}):
  ctx.template(tpl, Label("@bazel_tools//tools/cpp:%s.tpl" % tpl),
               substitutions)


def _impl(ctx):
  cpu_value = _get_cpu_value(ctx)
  darwin = cpu_value == "darwin"
  cc = _find_cc(ctx)
  crosstool_cc = "osx_cc_wrapper.sh" if darwin else str(cc)
  darwin = cpu_value == "darwin"
  tool_paths = _get_tool_paths(ctx, darwin, crosstool_cc)
  crosstool_content = _crosstool_content(ctx, cc, cpu_value, darwin)
  opt_content = _opt_content(darwin)
  dbg_content = _dbg_content()
  _tpl(ctx, "BUILD", {
      "%{name}": cpu_value,
      "%{supports_param_files}": "0" if darwin else "1"
  })
  _tpl(ctx, "osx_cc_wrapper.sh", {"%{cc}": str(cc)})
  _tpl(ctx, "CROSSTOOL", {
      "%{cpu}": cpu_value,
      "%{content}": _build_crosstool(crosstool_content) + "\n" +
                    _build_tool_path(tool_paths),
      "%{opt_content}": _build_crosstool(opt_content, "    "),
      "%{dbg_content}": _build_crosstool(dbg_content, "    "),
  })


cc_autoconf = repository_rule(_impl, local=True)


def cc_configure():
  """A C++ configuration rules that generate the crosstool file."""
  cc_autoconf(name="local_config_cc")
  native.bind(name="cc_toolchain", actual="@local_config_cc//:toolchain")
