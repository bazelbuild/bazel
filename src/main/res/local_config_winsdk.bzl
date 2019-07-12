# Copyright 2019 The Bazel Authors. All rights reserved.
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

load("//tools/cpp:windows_cc_configure.bzl", "find_vc_path", "setup_vc_env_vars")
load("//tools/cpp:cc_configure.bzl", "MSVC_ENVVARS")

def _find_rc_exe(repository_ctx):
    vc = find_vc_path(repository_ctx)
    if vc:
        env = setup_vc_env_vars(repository_ctx, vc, envvars = ["WindowsSdkVerBinPath"], escape = False)
        sdk = env.get("WindowsSdkVerBinPath")
        if sdk:
            exe = repository_ctx.path(sdk).get_child("x64").get_child("rc.exe")
            if exe.exists:
                return str(exe)
    return ""

def _impl(repository_ctx):
    rc_path = _find_rc_exe(repository_ctx)

    repository_ctx.file(
        "rc_exe.bat",
        content = ("@echo off\n\"%s\" %%*" % rc_path) if rc_path else ("@echo Could not find Windows SDK.\n@exit /b 1"),
        executable = True,
    )

    repository_ctx.file(
        "BUILD",
        content = """exports_files(["rc_exe.bat"])""",
        executable = False,
    )

local_config_winsdk = repository_rule(
    implementation = _impl,
    local = True,
    environ = list(MSVC_ENVVARS),
)
