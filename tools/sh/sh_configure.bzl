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
"""Configure sh_toolchains based on the local machine."""

_DEFAULT_SHELL_PATHS = {
    "windows": "c:/msys64/usr/bin/bash.exe",
    "linux": "/bin/bash",
    "osx": "/bin/bash",
    "freebsd": "/usr/local/bin/bash",
    "openbsd": "/usr/local/bin/bash",
}

_UNIX_SH_TOOLCHAIN_TEMPLATE = """
sh_toolchain(
    name = "{os}_sh",
    path = {sh_path},
)
"""

_WINDOWS_SH_TOOLCHAIN_TEMPLATE = """
sh_toolchain(
    name = "{os}_sh",
    path = {sh_path},
    launcher = "@bazel_tools//tools/launcher",
    launcher_maker = "@bazel_tools//tools/launcher:launcher_maker",
)
"""

_TOOLCHAIN_TEMPLATE = """
toolchain(
    name = "{os}_sh_toolchain",
    toolchain = ":{os}_sh",
    toolchain_type = "@bazel_tools//tools/sh:toolchain_type",
    target_compatible_with = [
        "@platforms//os:{os}",
    ],
)
"""

def _sh_config_impl(repository_ctx):
    """sh_config rule implementation.

    Creates sh_toolchains for commonly supported target platforms.
    For the target platform matching the local machine, it detects the path of
    the shell interpreter instead of using the default path.

    Args:
      repository_ctx: the repository rule context object
    """
    toolchains = []
    for os, default_shell_path in _DEFAULT_SHELL_PATHS.items():
        is_host = repository_ctx.os.name.startswith(os)
        if is_host:
            # This toolchain was first added before optional toolchains were
            # available, so instead of not registering a toolchain if we
            # couldn't find the shell, we register a toolchain with an empty
            # path.
            sh_path = _detect_local_shell_path(repository_ctx) or ""
        else:
            sh_path = default_shell_path

        sh_toolchain_template = _WINDOWS_SH_TOOLCHAIN_TEMPLATE if os == "windows" else _UNIX_SH_TOOLCHAIN_TEMPLATE
        toolchains.append(sh_toolchain_template.format(
            os = os,
            sh_path = repr(sh_path),
        ))
        toolchains.append(_TOOLCHAIN_TEMPLATE.format(
            os = os,
        ))

    repository_ctx.file("BUILD", """
load("@bazel_tools//tools/sh:sh_toolchain.bzl", "sh_toolchain")
""" + "\n".join(toolchains))

sh_config = repository_rule(
    environ = [
        "WINDIR",
        "PATH",
    ],
    # TODO: Replace this with configure = True and add BAZEL_SH to the
    # environ list above for consistency with CC and other repo rules.
    # This would make discovery differ from --shell_executable.
    local = True,
    implementation = _sh_config_impl,
)

def _detect_local_shell_path(repository_ctx):
    if repository_ctx.os.name.startswith("windows"):
        return _detect_local_shell_path_windows(repository_ctx)
    else:
        return _detect_local_shell_path_unix(repository_ctx)

def _detect_local_shell_path_windows(repository_ctx):
    sh_path = repository_ctx.os.environ.get("BAZEL_SH")
    if sh_path:
        return sh_path.replace("\\", "/")

    sh_path_obj = repository_ctx.which("bash.exe")
    if sh_path_obj:
        # repository_ctx.which returns a path object, convert that to
        # string so we can call string.startswith on it.
        sh_path = str(sh_path_obj)

        # When the Windows Subsystem for Linux is installed there's a
        # bash.exe under %WINDIR%\system32\bash.exe that launches Ubuntu
        # Bash which cannot run native Windows programs so it's not what
        # we want.
        windir = repository_ctx.os.environ.get("WINDIR")
        if not windir or not sh_path.startswith(windir):
            return sh_path.replace("\\", "/")

    return None

def _detect_local_shell_path_unix(repository_ctx):
    sh_path = repository_ctx.os.environ.get("BAZEL_SH")
    if sh_path:
        return sh_path

    sh_path_obj = repository_ctx.which("bash")
    if sh_path_obj:
        return str(sh_path_obj)

    sh_path_obj = repository_ctx.which("sh")
    if sh_path_obj:
        return str(sh_path_obj)

    return None

def sh_configure():
    """Detect the local shell interpreter and register its toolchain."""
    sh_config(name = "local_config_sh")
    native.register_toolchains("@local_config_sh//:all")

def _sh_configure_extension_impl(module_ctx):
    sh_config(name = "local_config_sh")
    return module_ctx.extension_metadata(reproducible = True)

sh_configure_extension = module_extension(implementation = _sh_configure_extension_impl)
