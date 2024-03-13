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
Definition of java_toolchain rule and JavaToolchainInfo provider.
"""

load(":common/java/boot_class_path_info.bzl", "BootClassPathInfo")
load(":common/java/java_helper.bzl", "helper")
load(":common/java/java_info.bzl", "JavaPluginDataInfo")
load(":common/java/java_package_configuration.bzl", "JavaPackageConfigurationInfo")
load(":common/java/java_runtime.bzl", "JavaRuntimeInfo")
load(":common/java/java_semantics.bzl", "semantics")

_java_common_internal = _builtins.internal.java_common_internal_do_not_use
ToolchainInfo = _builtins.toplevel.platform_common.ToolchainInfo
PackageSpecificationInfo = _builtins.toplevel.PackageSpecificationInfo

def _java_toolchain_info_init(**_kwargs):
    fail("JavaToolchainInfo instantiation is a private API")

_PRIVATE_API_DOC_STRING = "internal API, DO NOT USE!"

JavaToolchainInfo, _new_javatoolchaininfo = provider(
    doc = "Information about the JDK used by the <code>java_*</code> rules.",
    fields = {
        "bootclasspath": "(depset[File]) The Java target bootclasspath entries. Corresponds to javac's -bootclasspath flag.",
        "ijar": "(FilesToRunProvider) The ijar executable.",
        "jacocorunner": "(FilesToRunProvider) The jacocorunner used by the toolchain.",
        "java_runtime": "(JavaRuntimeInfo) The java runtime information.",
        "jvm_opt": "(depset[str]) The default options for the JVM running the java compiler and associated tools.",
        "label": "(label) The toolchain label.",
        "proguard_allowlister": "(FilesToRunProvider) The binary to validate proguard configuration.",
        "single_jar": "(FilesToRunProvider) The SingleJar deploy jar.",
        "source_version": "(str) The java source version.",
        "target_version": "(str) The java target version.",
        "tools": "(depset[File]) The compilation tools.",
        # private
        "_android_linter": _PRIVATE_API_DOC_STRING,
        "_bootclasspath_info": _PRIVATE_API_DOC_STRING,
        "_bytecode_optimizer": _PRIVATE_API_DOC_STRING,
        "_compatible_javacopts": _PRIVATE_API_DOC_STRING,
        "_deps_checker": _PRIVATE_API_DOC_STRING,
        "_forcibly_disable_header_compilation": _PRIVATE_API_DOC_STRING,
        "_gen_class": _PRIVATE_API_DOC_STRING,
        "_header_compiler": _PRIVATE_API_DOC_STRING,
        "_header_compiler_builtin_processors": _PRIVATE_API_DOC_STRING,
        "_header_compiler_direct": _PRIVATE_API_DOC_STRING,
        "_javabuilder": _PRIVATE_API_DOC_STRING,
        "_javacopts": _PRIVATE_API_DOC_STRING,
        "_javac_supports_workers": _PRIVATE_API_DOC_STRING,
        "_javac_supports_multiplex_workers": _PRIVATE_API_DOC_STRING,
        "_javac_supports_worker_cancellation": _PRIVATE_API_DOC_STRING,
        "_javac_supports_worker_multiplex_sandboxing": _PRIVATE_API_DOC_STRING,
        "_jspecify_info": _PRIVATE_API_DOC_STRING,
        "_local_java_optimization_config": _PRIVATE_API_DOC_STRING,
        "_one_version_tool": _PRIVATE_API_DOC_STRING,
        "_one_version_allowlist": _PRIVATE_API_DOC_STRING,
        "_one_version_allowlist_for_tests": _PRIVATE_API_DOC_STRING,
        "_package_configuration": _PRIVATE_API_DOC_STRING,
        "_reduced_classpath_incompatible_processors": _PRIVATE_API_DOC_STRING,
        "_timezone_data": _PRIVATE_API_DOC_STRING,
    },
    init = _java_toolchain_info_init,
)

def _java_toolchain_impl(ctx):
    bootclasspath_info = _get_bootclasspath_info(ctx)
    java_runtime = _get_java_runtime(ctx)
    if java_runtime and java_runtime.lib_ct_sym:
        header_compiler_direct_data = [java_runtime.lib_ct_sym]
        header_compiler_direct_jvm_opts = ["-Dturbine.ctSymPath=" + java_runtime.lib_ct_sym.path]
    elif java_runtime and java_runtime.java_home:
        # Turbine finds ct.sym relative to java.home.
        header_compiler_direct_data = []
        header_compiler_direct_jvm_opts = ["-Djava.home=" + java_runtime.java_home]
    else:
        header_compiler_direct_data = []
        header_compiler_direct_jvm_opts = []
    java_toolchain_info = _new_javatoolchaininfo(
        bootclasspath = bootclasspath_info.bootclasspath,
        ijar = ctx.attr.ijar.files_to_run if ctx.attr.ijar else None,
        jacocorunner = ctx.attr.jacocorunner.files_to_run if ctx.attr.jacocorunner else None,
        java_runtime = java_runtime,
        jvm_opt = depset(_java_common_internal.expand_java_opts(ctx, "jvm_opts", tokenize = False, exec_paths = True)),
        label = ctx.label,
        proguard_allowlister = ctx.attr.proguard_allowlister.files_to_run if ctx.attr.proguard_allowlister else None,
        single_jar = ctx.attr.singlejar.files_to_run,
        source_version = ctx.attr.source_version,
        target_version = ctx.attr.target_version,
        tools = depset(ctx.files.tools),
        # private
        _android_linter = _get_android_lint_tool(ctx),
        _bootclasspath_info = bootclasspath_info,
        _bytecode_optimizer = _get_tool_from_executable(ctx, "_bytecode_optimizer"),
        _compatible_javacopts = _get_compatible_javacopts(ctx),
        _deps_checker = ctx.file.deps_checker,
        _forcibly_disable_header_compilation = ctx.attr.forcibly_disable_header_compilation,
        _gen_class = ctx.file.genclass,
        _header_compiler = _get_tool_from_ctx(ctx, "header_compiler", "turbine_data", "turbine_jvm_opts"),
        _header_compiler_builtin_processors = depset(ctx.attr.header_compiler_builtin_processors),
        _header_compiler_direct = _get_tool_from_executable(
            ctx,
            "header_compiler_direct",
            data = header_compiler_direct_data,
            jvm_opts = header_compiler_direct_jvm_opts,
        ),
        _javabuilder = _get_tool_from_ctx(ctx, "javabuilder", "javabuilder_data", "javabuilder_jvm_opts"),
        _javacopts = _get_javac_opts(ctx),
        _javac_supports_workers = ctx.attr.javac_supports_workers,
        _javac_supports_multiplex_workers = ctx.attr.javac_supports_multiplex_workers,
        _javac_supports_worker_cancellation = ctx.attr.javac_supports_worker_cancellation,
        _javac_supports_worker_multiplex_sandboxing = ctx.attr.javac_supports_worker_multiplex_sandboxing,
        _jspecify_info = _get_jspecify_info(ctx),
        _local_java_optimization_config = ctx.files._local_java_optimization_configuration,
        _one_version_tool = ctx.attr.oneversion.files_to_run if ctx.attr.oneversion else None,
        _one_version_allowlist = ctx.file.oneversion_whitelist,
        _one_version_allowlist_for_tests = ctx.file.oneversion_allowlist_for_tests,
        _package_configuration = [dep[JavaPackageConfigurationInfo] for dep in ctx.attr.package_configuration],
        _reduced_classpath_incompatible_processors = depset(ctx.attr.reduced_classpath_incompatible_processors, order = "preorder"),
        _timezone_data = ctx.file.timezone_data,
    )
    toolchain_info = ToolchainInfo(java = java_toolchain_info)
    return [java_toolchain_info, toolchain_info, DefaultInfo()]

def _get_bootclasspath_info(ctx):
    bootclasspath_infos = [dep[BootClassPathInfo] for dep in ctx.attr.bootclasspath if BootClassPathInfo in dep]
    if bootclasspath_infos:
        if len(bootclasspath_infos) != 1:
            fail("in attribute 'bootclasspath': expected exactly one entry with a BootClassPathInfo provider")
        else:
            return bootclasspath_infos[0]
    else:
        return BootClassPathInfo(bootclasspath = ctx.files.bootclasspath)

def _get_java_runtime(ctx):
    if not ctx.attr.java_runtime:
        return None
    return ctx.attr.java_runtime[ToolchainInfo].java_runtime

def _get_javac_opts(ctx):
    opts = []
    if ctx.attr.source_version:
        opts.extend(["-source", ctx.attr.source_version])
    if ctx.attr.target_version:
        opts.extend(["-target", ctx.attr.target_version])
    if ctx.attr.xlint:
        opts.append("-Xlint:" + ",".join(ctx.attr.xlint))
    opts.extend(_java_common_internal.expand_java_opts(ctx, "misc", tokenize = True))
    opts.extend(_java_common_internal.expand_java_opts(ctx, "javacopts", tokenize = True))
    return helper.detokenize_javacopts(opts)

def _get_android_lint_tool(ctx):
    if not ctx.attr.android_lint_runner:
        return None
    files_to_run = ctx.attr.android_lint_runner.files_to_run
    if not files_to_run or not files_to_run.executable:
        fail(ctx.attr.android_lint_runner.label, "does not refer to a valid executable target")
    return struct(
        tool = files_to_run,
        data = depset(ctx.files.android_lint_data),
        jvm_opts = depset([ctx.expand_location(opt, ctx.attr.android_lint_data) for opt in ctx.attr.android_lint_jvm_opts]),
        lint_opts = [ctx.expand_location(opt, ctx.attr.android_lint_data) for opt in ctx.attr.android_lint_opts],
        package_config = [dep[JavaPackageConfigurationInfo] for dep in ctx.attr.android_lint_package_configuration],
    )

def _get_tool_from_ctx(ctx, tool_attr, data_attr, opts_attr):
    dep = getattr(ctx.attr, tool_attr)
    if not dep:
        return None
    files_to_run = dep.files_to_run
    if not files_to_run or not files_to_run.executable:
        fail(dep.label, "does not refer to a valid executable target")
    data = getattr(ctx.attr, data_attr)
    return struct(
        tool = files_to_run,
        data = depset(getattr(ctx.files, data_attr)),
        jvm_opts = depset([ctx.expand_location(opt, data) for opt in getattr(ctx.attr, opts_attr)]),
    )

def _get_tool_from_executable(ctx, attr_name, data = [], jvm_opts = []):
    dep = getattr(ctx.attr, attr_name)
    if not dep:
        return None
    files_to_run = dep.files_to_run
    if not files_to_run or not files_to_run.executable:
        fail(dep.label, "does not refer to a valid executable target")
    return struct(tool = files_to_run, data = depset(data), jvm_opts = depset(jvm_opts))

def _get_compatible_javacopts(ctx):
    result = {}
    for key, opt_list in ctx.attr.compatible_javacopts.items():
        result[key] = helper.detokenize_javacopts([token for opt in opt_list for token in ctx.tokenize(opt)])
    return result

def _get_jspecify_info(ctx):
    if not ctx.attr.jspecify_processor_class:
        return None
    stubs = ctx.files.jspecify_stubs
    javacopts = []
    javacopts.extend(ctx.attr.jspecify_javacopts)
    if stubs:
        javacopts.append("-Astubs=" + ":".join([file.path for file in stubs]))
    return struct(
        processor = JavaPluginDataInfo(
            processor_classes = depset([ctx.attr.jspecify_processor_class]),
            processor_jars = depset([ctx.file.jspecify_processor]),
            processor_data = depset(stubs),
        ),
        implicit_deps = depset([ctx.file.jspecify_implicit_deps]),
        javacopts = javacopts,
        packages = [target[PackageSpecificationInfo] for target in ctx.attr.jspecify_packages],
    )

_java_toolchain = rule(
    implementation = _java_toolchain_impl,
    attrs = {
        "android_lint_data": attr.label_list(cfg = "exec", allow_files = True),
        "android_lint_opts": attr.string_list(default = []),
        "android_lint_jvm_opts": attr.string_list(default = []),
        "android_lint_package_configuration": attr.label_list(cfg = "exec", providers = [JavaPackageConfigurationInfo], allow_files = True),
        "android_lint_runner": attr.label(cfg = "exec", executable = True, allow_single_file = True),
        "bootclasspath": attr.label_list(default = [], allow_files = True),
        "compatible_javacopts": attr.string_list_dict(),
        "deps_checker": attr.label(allow_single_file = True, cfg = "exec", executable = True),
        "forcibly_disable_header_compilation": attr.bool(default = False),
        "genclass": attr.label(allow_single_file = True, cfg = "exec", executable = True),
        "header_compiler": attr.label(allow_single_file = True, cfg = "exec", executable = True),
        "header_compiler_direct": attr.label(allow_single_file = True, cfg = "exec", executable = True),
        "header_compiler_builtin_processors": attr.string_list(),
        "ijar": attr.label(cfg = "exec", allow_files = True, executable = True),
        "jacocorunner": attr.label(cfg = "exec", allow_single_file = True, executable = True),
        "javabuilder": attr.label(cfg = "exec", allow_single_file = True, executable = True),
        "javabuilder_data": attr.label_list(cfg = "exec", allow_files = True),
        "javabuilder_jvm_opts": attr.string_list(),
        "java_runtime": attr.label(cfg = "exec", providers = [JavaRuntimeInfo]),
        "javac_supports_workers": attr.bool(default = True),
        "javac_supports_multiplex_workers": attr.bool(default = True),
        "javac_supports_worker_cancellation": attr.bool(default = True),
        "javac_supports_worker_multiplex_sandboxing": attr.bool(default = False),
        "javacopts": attr.string_list(default = []),
        "jspecify_implicit_deps": attr.label(cfg = "exec", allow_single_file = True, executable = True),
        "jspecify_javacopts": attr.string_list(),
        "jspecify_packages": attr.label_list(cfg = "exec", allow_files = True, providers = [PackageSpecificationInfo]),
        "jspecify_processor": attr.label(cfg = "exec", allow_single_file = True, executable = True),
        "jspecify_processor_class": attr.string(),
        "jspecify_stubs": attr.label_list(cfg = "exec", allow_files = True),
        "jvm_opts": attr.string_list(default = []),
        "misc": attr.string_list(default = []),
        "oneversion": attr.label(cfg = "exec", allow_files = True, executable = True),
        "oneversion_whitelist": attr.label(allow_single_file = True),
        "oneversion_allowlist_for_tests": attr.label(allow_single_file = True),
        "package_configuration": attr.label_list(cfg = "exec", providers = [JavaPackageConfigurationInfo]),
        "proguard_allowlister": attr.label(cfg = "exec", executable = True, allow_files = True, default = semantics.PROGUARD_ALLOWLISTER_LABEL),
        "reduced_classpath_incompatible_processors": attr.string_list(),
        "singlejar": attr.label(cfg = "exec", allow_files = True, executable = True),
        "source_version": attr.string(),
        "target_version": attr.string(),
        "timezone_data": attr.label(cfg = "exec", allow_single_file = True),
        "tools": attr.label_list(cfg = "exec", allow_files = True),
        "turbine_data": attr.label_list(cfg = "exec", allow_files = True),
        "turbine_jvm_opts": attr.string_list(),
        "xlint": attr.string_list(default = []),
        "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
        "_bytecode_optimizer": attr.label(
            cfg = "exec",
            executable = True,
            default = configuration_field(fragment = "java", name = "java_toolchain_bytecode_optimizer"),
        ),
        "_local_java_optimization_configuration": attr.label(
            cfg = "exec",
            default = configuration_field(fragment = "java", name = "local_java_optimization_configuration"),
            allow_files = True,
        ),
    },
    fragments = ["java"],
)

def _extract_singleton_list_value(dict, key):
    if key in dict and type(dict[key]) == type([]):
        list = dict[key]
        if len(list) > 1:
            fail("expected a single value for:", key, "got: ", list)
        elif len(list) == 1:
            dict[key] = dict[key][0]
        else:
            dict[key] = None

def _java_toolchain_macro(**kwargs):
    # these attributes are defined as executable `label_list`s in native but are
    # expected to be singleton values. Since this is not supported in Starlark,
    # we just inline the value from the list (if present) before invoking the
    # rule.
    _extract_singleton_list_value(kwargs, "genclass")
    _extract_singleton_list_value(kwargs, "deps_checker")
    _extract_singleton_list_value(kwargs, "header_compiler")
    _extract_singleton_list_value(kwargs, "header_compiler_direct")
    _extract_singleton_list_value(kwargs, "ijar")
    _extract_singleton_list_value(kwargs, "javabuilder")
    _extract_singleton_list_value(kwargs, "singlejar")
    _java_toolchain(**kwargs)

java_toolchain = _java_toolchain_macro
