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

def _is_windows(repository_ctx):
    """Returns true if the host OS is Windows."""
    return repository_ctx.os.name.startswith("windows")

def _sh_config_impl(repository_ctx):
    """sh_config rule implementation.

    Detects the path of the shell interpreter on the local machine and
    stores it in a sh_toolchain rule.

    Args:
      repository_ctx: the repository rule context object
    """
    sh_path = repository_ctx.os.environ.get("BAZEL_SH")
    if not sh_path:
        if _is_windows(repository_ctx):
            sh_path = repository_ctx.which("bash.exe")
            if sh_path:
                # repository_ctx.which returns a path object, convert that to
                # string so we can call string.startswith on it.
                sh_path = str(sh_path)

                # When the Windows Subsystem for Linux is installed there's a
                # bash.exe under %WINDIR%\system32\bash.exe that launches Ubuntu
                # Bash which cannot run native Windows programs so it's not what
                # we want.
                windir = repository_ctx.os.environ.get("WINDIR")
                if windir and sh_path.startswith(windir):
                    sh_path = None
        else:
            sh_path = repository_ctx.which("bash")
            if not sh_path:
                sh_path = repository_ctx.which("sh")

    if not sh_path:
        sh_path = ""

    if sh_path and _is_windows(repository_ctx):
        sh_path = sh_path.replace("\\", "/")

    repository_ctx.file("BUILD", """
load("@bazel_tools//tools/sh:sh_toolchain.bzl", "sh_toolchain")

sh_toolchain(
    name = "local_sh",
    path = "{sh_path}",
    visibility = ["//visibility:public"],
)

toolchain(
    name = "local_sh_toolchain",
    toolchain = ":local_sh",
    toolchain_type = "@bazel_tools//tools/sh:toolchain_type",
)
""".format(sh_path = sh_path))

sh_config = repository_rule(
    environ = [
        "WINDIR",
        "PATH",
    ],
    local = True,
    implementation = _sh_config_impl,
)

def sh_configure():
    """Detect the local shell interpreter and register its toolchain."""
    sh_config(name = "local_config_sh")
    native.register_toolchains("@local_config_sh//:local_sh_toolchain")
