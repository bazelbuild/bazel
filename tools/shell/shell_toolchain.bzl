# Copyright 2018 The Bazel Authors. All rights reserved.
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
"""Configure the shell toolchain on the local machine."""

def _shell_toolchain_impl(ctx):
  """shell_toolchain rule implementation."""
  return [platform_common.ToolchainInfo(path = ctx.attr.path)]

def _is_windows(repository_ctx):
  """Returns true if the host OS is Windows."""
  return repository_ctx.os.name.startswith("windows")

def _shell_config_impl(repository_ctx):
  """shell_config rule implementation.

  Detects the path of the shell interpreter on the local machine and
  stores it in a shell_toolchain rule.
  """
  shell_path = repository_ctx.os.environ.get("BAZEL_SH")
  if not shell_path:
    if _is_windows(repository_ctx):
      shell_path = repository_ctx.which("bash.exe")
      if shell_path:
        # When the Windows Subsystem for Linux is installed there's a
        # bash.exe under %WINDIR%\system32\bash.exe that launches Ubuntu
        # Bash which cannot run native Windows programs so it's not what
        # we want.
        windir = repository_ctx.os.environ.get("WINDIR")
        if windir and shell_path.startswith(windir):
          shell_path = None
    else:
      shell_path = repository_ctx.which("bash")
      if not shell_path:
        shell_path = repository_ctx.which("sh")

  if not shell_path:
    shell_path = ""

  if shell_path and _is_windows(repository_ctx):
    shell_path = shell_path.replace("\\", "/")

  os_label = None
  if _is_windows(repository_ctx):
    os_label = "@bazel_tools//platforms:windows"
  elif repository_ctx.os.name.startswith("linux"):
    os_label = "@bazel_tools//platforms:linux"
  elif repository_ctx.os.name.startswith("mac"):
    os_label = "@bazel_tools//platforms:osx"
  else:
    fail("Unknown OS")

  repository_ctx.file("BUILD", """
load("@bazel_tools//tools/shell:shell_toolchain.bzl", "shell_toolchain")

shell_toolchain(
    name = "local_shell",
    path = "{shell_path}",
    visibility = ["//visibility:public"],
)

toolchain(
    name = "local_shell_toolchain",
    exec_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "{os_label}",
    ],
    toolchain = ":local_shell",
    toolchain_type = "@bazel_tools//tools/shell:toolchain_type",
)
""".format(shell_path = shell_path, os_label = os_label))

shell_toolchain = rule(
    attrs = {"path": attr.string()},
    implementation = _shell_toolchain_impl,
)

shell_config = repository_rule(
    environ = [
        "WINDIR",
        "PATH",
    ],
    local = True,
    implementation = _shell_config_impl,
)

def shell_repositories():
  """Detect the local shell interpreter and register its toolchain."""
  shell_config(name = "local_config_shell")
  native.register_toolchains("@local_config_shell//:local_shell_toolchain")
