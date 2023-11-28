# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""
Definition of java_runtime rule and JavaRuntimeInfo provider.
"""

load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/paths.bzl", "paths")

platform_common = _builtins.toplevel.platform_common
ToolchainInfo = _builtins.toplevel.platform_common.ToolchainInfo

_java_common_internal = _builtins.internal.java_common_internal_do_not_use

def _init_java_runtime_info(**_kwargs):
    fail("instantiating JavaRuntimeInfo is a private API")

JavaRuntimeInfo, _new_javaruntimeinfo = provider(
    doc = "Information about the Java runtime used by the java rules.",
    fields = {
        "default_cds": "Returns the JDK default CDS archive.",
        "files": "Returns the files in the Java runtime.",
        "hermetic_files": "Returns the files in the Java runtime needed for hermetic deployments.",
        "hermetic_static_libs": "Returns the JDK static libraries.",
        "java_executable_exec_path": "Returns the execpath of the Java executable.",
        "java_executable_runfiles_path": """Returns the path of the Java executable in
                runfiles trees. This should only be used when one needs to access the
                JVM during the execution of a binary or a test built by Bazel. In particular,
                when one needs to invoke the JVM during an action, java_executable_exec_path
                should be used instead.""",
        "java_home": "Returns the execpath of the root of the Java installation.",
        "java_home_runfiles_path": """Returns the path of the Java installation in runfiles trees.
                This should only be used when one needs to access the JDK during the execution
                of a binary or a test built by Bazel. In particular, when one needs the JDK
                during an action, java_home should be used instead.""",
        "lib_ct_sym": "Returns the lib/ct.sym file.",
        "lib_modules": "Returns the lib/modules file.",
        "version": "The Java feature version of the runtime. This is 0 if the version is unknown.",
    },
    init = _init_java_runtime_info,
)

def _is_main_repo(label):
    return label.workspace_name == ""

def _default_java_home(label):
    if _is_main_repo(label):
        return label.package
    else:
        return paths.get_relative(label.workspace_root, label.package)

def _get_bin_java():
    #TODO: b/304990922 - use platforms instead of checking host
    is_windows = _java_common_internal.current_os_name == "windows"
    return "bin/java.exe" if is_windows else "bin/java"

def _get_runfiles_java_executable(java_home, label):
    if paths.is_absolute(java_home) or _is_main_repo(label):
        return paths.get_relative(java_home, _get_bin_java())
    else:
        repo_runfiles_path = "" if _is_main_repo(label) else paths.get_relative("..", label.workspace_name)
        return paths.get_relative(repo_runfiles_path, _get_bin_java())

def _is_java_binary(path):
    return path.endswith("bin/java") or path.endswith("bin/java.exe")

def _get_lib_ct_sym(srcs, explicit_lib_ct_sym):
    if explicit_lib_ct_sym:
        return explicit_lib_ct_sym
    candidates = [src for src in srcs if src.path.endswith("/lib/ct.sym")]
    if len(candidates) == 1:
        return candidates[0]
    else:
        return None

def _java_runtime_rule_impl(ctx):
    all_files = []  # [depset[File]]
    all_files.append(depset(ctx.files.srcs))

    java_home = _default_java_home(ctx.label)
    if ctx.attr.java_home:
        java_home_attr = ctx.expand_make_variables("java_home", ctx.attr.java_home, {})
        if ctx.files.srcs and paths.is_absolute(java_home_attr):
            fail("'java_home' with an absolute path requires 'srcs' to be empty.")
        java_home = paths.get_relative(java_home, java_home_attr)

    java_binary_exec_path = paths.get_relative(java_home, _get_bin_java())
    java_binary_runfiles_path = _get_runfiles_java_executable(java_home, ctx.label)

    java = ctx.file.java
    if java:
        if paths.is_absolute(java_home):
            fail("'java_home' with an absolute path requires 'java' to be empty.")
        java_binary_exec_path = java.path
        java_binary_runfiles_path = java.short_path
        if not _is_java_binary(java_binary_exec_path):
            fail("the path to 'java' must end in 'bin/java'.")
        java_home = paths.dirname(paths.dirname(java_binary_exec_path))
        all_files.append(depset([java]))

    java_home_runfiles_path = paths.dirname(paths.dirname(java_binary_runfiles_path))

    hermetic_inputs = depset(ctx.files.hermetic_srcs)
    all_files.append(hermetic_inputs)

    lib_ct_sym = _get_lib_ct_sym(ctx.files.srcs, ctx.file.lib_ct_sym)
    lib_modules = ctx.file.lib_modules
    hermetic_static_libs = [dep[CcInfo] for dep in ctx.attr.hermetic_static_libs]

    # If a runtime does not set default_cds in hermetic mode, it is not fatal.
    # We can skip the default CDS in the check below.
    default_cds = ctx.file.default_cds

    if (hermetic_inputs or lib_modules or hermetic_static_libs) and (
        not hermetic_inputs or not lib_modules or not hermetic_static_libs
    ):
        fail("hermetic specified, all of java_runtime.lib_modules, java_runtime.hermetic_srcs and java_runtime.hermetic_static_libs must be specified")

    files = depset(transitive = all_files)

    java_runtime_info = _new_javaruntimeinfo(
        default_cds = default_cds,
        files = files,
        hermetic_files = hermetic_inputs,
        hermetic_static_libs = hermetic_static_libs,
        java_executable_exec_path = java_binary_exec_path,
        java_executable_runfiles_path = java_binary_runfiles_path,
        java_home = java_home,
        java_home_runfiles_path = java_home_runfiles_path,
        lib_ct_sym = lib_ct_sym,
        lib_modules = lib_modules,
        version = ctx.attr.version,
    )
    return [
        DefaultInfo(
            files = files,
            runfiles = ctx.runfiles(transitive_files = files),
        ),
        java_runtime_info,
        platform_common.TemplateVariableInfo({
            "JAVA": java_binary_exec_path,
            "JAVABASE": java_home,
        }),
        ToolchainInfo(java_runtime = java_runtime_info),
    ]

java_runtime = rule(
    implementation = _java_runtime_rule_impl,
    attrs = {
        "default_cds": attr.label(allow_single_file = True, executable = True, cfg = "target"),
        "hermetic_srcs": attr.label_list(allow_files = True),
        "hermetic_static_libs": attr.label_list(providers = [CcInfo]),
        "java": attr.label(allow_single_file = True, executable = True, cfg = "target"),
        "java_home": attr.string(),
        "lib_ct_sym": attr.label(allow_single_file = True),
        "lib_modules": attr.label(allow_single_file = True, executable = True, cfg = "target"),
        "output_licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
        "srcs": attr.label_list(allow_files = True),
        "version": attr.int(),
    },
    fragments = ["java"],
    provides = [
        JavaRuntimeInfo,
        platform_common.TemplateVariableInfo,
    ],
)
