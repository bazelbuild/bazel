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
"""Configure the Bash toolchain on the local machine."""

def _bash_toolchain_impl(ctx):
  """bash_toolchain rule implementation."""
  return [platform_common.ToolchainInfo(path = ctx.attr.path)]

def _is_windows(repository_ctx):
  """Returns true if the host OS is Windows."""
  return repository_ctx.os.name.startswith("windows")

def _bash_config_impl(repository_ctx):
  """bash_config rule implementation.

  Detects the path of Bash on the local machine and stores it in a
  bash_toolchain rule.
  """
  bash_path = repository_ctx.os.environ.get("BAZEL_SH")
  if not bash_path:
    if _is_windows(repository_ctx):
      bash_path = repository_ctx.which("bash.exe")
      if bash_path:
        # When the Windows Subsystem for Linux is installed there's a
        # bash.exe under %WINDIR%\system32\bash.exe that launches Ubuntu
        # Bash which cannot run native Windows programs so it's not what
        # we want.
        windir = repository_ctx.os.environ.get("WINDIR")
        if windir and bash_path.startswith(windir):
          bash_path = None
    else:
      bash_path = repository_ctx.which("bash")

  if not bash_path:
    bash_path = ""

  if bash_path and _is_windows(repository_ctx):
    bash_path = bash_path.replace("\\", "/")

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
load("@bazel_tools//tools/bash:bash_def.bzl", "bash_toolchain")

bash_toolchain(
    name = "local_bash",
    path = "{bash_path}",
    visibility = ["//visibility:public"],
)

toolchain(
    name = "local_bash_toolchain",
    exec_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "{os_label}",
    ],
    toolchain = ":local_bash",
    toolchain_type = "@bazel_tools//tools/bash:bash_toolchain_type",
)
""".format(bash_path = bash_path, os_label = os_label))

bash_toolchain = rule(
    attrs = {"path": attr.string()},
    implementation = _bash_toolchain_impl,
)

bash_config = repository_rule(
    environ = [
        "WINDIR",
        "PATH",
    ],
    local = True,
    implementation = _bash_config_impl,
)

def bash_repositories():
  """Detect the local Bash and register it as a toolchain."""
  bash_config(name = "local_config_bash")
  native.register_toolchains("@local_config_bash//:local_bash_toolchain")
