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
        "_javacopts_list": _PRIVATE_API_DOC_STRING,
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
    javac_opts_list = _get_javac_opts(ctx)
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
        _javacopts = helper.detokenize_javacopts(javac_opts_list),
        _javacopts_list = javac_opts_list,
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
    return opts

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

def _extract_singleton_list_value(dict, key):
    if key in dict and type(dict[key]) == type([]):
        list = dict[key]
        if len(list) > 1:
            fail("expected a single value for:", key, "got: ", list)
        elif len(list) == 1:
            dict[key] = dict[key][0]
        else:
            dict[key] = None

_LEGACY_ANY_TYPE_ATTRS = [
    "genclass",
    "deps_checker",
    "header_compiler",
    "header_compiler_direct",
    "ijar",
    "javabuilder",
    "singlejar",
]

def _java_toolchain_initializer(**kwargs):
    # these attributes are defined as executable `label_list`s in native but are
    # expected to be singleton values. Since this is not supported in Starlark,
    # we just inline the value from the list (if present) before invoking the
    # rule.
    for attr in _LEGACY_ANY_TYPE_ATTRS:
        _extract_singleton_list_value(kwargs, attr)

    return kwargs

java_toolchain = rule(
    implementation = _java_toolchain_impl,
    initializer = _java_toolchain_initializer,
    doc = """
<p>
Specifies the configuration for the Java compiler. Which toolchain to be used can be changed through
the --java_toolchain argument. Normally you should not write those kind of rules unless you want to
tune your Java compiler.
</p>

<h4>Examples</h4>

<p>A simple example would be:
</p>

<pre class="code">
<code class="lang-starlark">

java_toolchain(
    name = "toolchain",
    source_version = "7",
    target_version = "7",
    bootclasspath = ["//tools/jdk:bootclasspath"],
    xlint = [ "classfile", "divzero", "empty", "options", "path" ],
    javacopts = [ "-g" ],
    javabuilder = ":JavaBuilder_deploy.jar",
)
</code>
</pre>
    """,
    attrs = {
        "android_lint_data": attr.label_list(
            cfg = "exec",
            allow_files = True,
            doc = """
Labels of tools available for label-expansion in android_lint_jvm_opts.
            """,
        ),
        "android_lint_opts": attr.string_list(
            default = [],
            doc = """
The list of Android Lint arguments.
            """,
        ),
        "android_lint_jvm_opts": attr.string_list(
            default = [],
            doc = """
The list of arguments for the JVM when invoking Android Lint.
            """,
        ),
        "android_lint_package_configuration": attr.label_list(
            cfg = "exec",
            providers = [JavaPackageConfigurationInfo],
            allow_files = True,
            doc = """
Android Lint Configuration that should be applied to the specified package groups.
            """,
        ),
        "android_lint_runner": attr.label(
            cfg = "exec",
            executable = True,
            allow_single_file = True,
            doc = """
Label of the Android Lint runner, if any.
            """,
        ),
        "bootclasspath": attr.label_list(
            default = [],
            allow_files = True,
            doc = """
The Java target bootclasspath entries. Corresponds to javac's -bootclasspath flag.
            """,
        ),
        "compatible_javacopts": attr.string_list_dict(
            doc = """Internal API, do not use!""",
        ),
        "deps_checker": attr.label(
            allow_single_file = True,
            cfg = "exec",
            executable = True,
            doc = """
Label of the ImportDepsChecker deploy jar.
            """,
        ),
        "forcibly_disable_header_compilation": attr.bool(
            default = False,
            doc = """
Overrides --java_header_compilation to disable header compilation on platforms that do not
support it, e.g. JDK 7 Bazel.
            """,
        ),
        "genclass": attr.label(
            allow_single_file = True,
            cfg = "exec",
            executable = True,
            doc = """
Label of the GenClass deploy jar.
            """,
        ),
        "header_compiler": attr.label(
            allow_single_file = True,
            cfg = "exec",
            executable = True,
            doc = """
Label of the header compiler. Required if --java_header_compilation is enabled.
            """,
        ),
        "header_compiler_direct": attr.label(
            allow_single_file = True,
            cfg = "exec",
            executable = True,
            doc = """
Optional label of the header compiler to use for direct classpath actions that do not
include any API-generating annotation processors.

<p>This tool does not support annotation processing.
            """,
        ),
        "header_compiler_builtin_processors": attr.string_list(
            doc = """Internal API, do not use!""",
        ),
        "ijar": attr.label(
            cfg = "exec",
            allow_files = True,
            executable = True,
            doc = """
Label of the ijar executable.
            """,
        ),
        "jacocorunner": attr.label(
            cfg = "exec",
            allow_single_file = True,
            executable = True,
            doc = """
Label of the JacocoCoverageRunner deploy jar.
            """,
        ),
        "javabuilder": attr.label(
            cfg = "exec",
            allow_single_file = True,
            executable = True,
            doc = """
Label of the JavaBuilder deploy jar.
            """,
        ),
        "javabuilder_data": attr.label_list(
            cfg = "exec",
            allow_files = True,
            doc = """
Labels of data available for label-expansion in javabuilder_jvm_opts.
            """,
        ),
        "javabuilder_jvm_opts": attr.string_list(
            doc = """
The list of arguments for the JVM when invoking JavaBuilder.
            """,
        ),
        "java_runtime": attr.label(
            cfg = "exec",
            providers = [JavaRuntimeInfo],
            doc = """
The java_runtime to use with this toolchain. It defaults to java_runtime
in execution configuration.
            """,
        ),
        "javac_supports_workers": attr.bool(
            default = True,
            doc = """
True if JavaBuilder supports running as a persistent worker, false if it doesn't.
            """,
        ),
        "javac_supports_multiplex_workers": attr.bool(
            default = True,
            doc = """
True if JavaBuilder supports running as a multiplex persistent worker, false if it doesn't.
            """,
        ),
        "javac_supports_worker_cancellation": attr.bool(
            default = True,
            doc = """
True if JavaBuilder supports cancellation of persistent workers, false if it doesn't.
            """,
        ),
        "javac_supports_worker_multiplex_sandboxing": attr.bool(
            default = False,
            doc = """
True if JavaBuilder supports running as a multiplex persistent worker with sandboxing, false if it doesn't.
            """,
        ),
        "javacopts": attr.string_list(
            default = [],
            doc = """
The list of extra arguments for the Java compiler. Please refer to the Java compiler
documentation for the extensive list of possible Java compiler flags.
            """,
        ),
        "jspecify_implicit_deps": attr.label(
            cfg = "exec",
            allow_single_file = True,
            executable = True,
            doc = """Experimental, do not use!""",
        ),
        "jspecify_javacopts": attr.string_list(
            doc = """Experimental, do not use!""",
        ),
        "jspecify_packages": attr.label_list(
            cfg = "exec",
            allow_files = True,
            providers = [PackageSpecificationInfo],
            doc = """Experimental, do not use!""",
        ),
        "jspecify_processor": attr.label(
            cfg = "exec",
            allow_single_file = True,
            executable = True,
            doc = """Experimental, do not use!""",
        ),
        "jspecify_processor_class": attr.string(
            doc = """Experimental, do not use!""",
        ),
        "jspecify_stubs": attr.label_list(
            cfg = "exec",
            allow_files = True,
            doc = """Experimental, do not use!""",
        ),
        "jvm_opts": attr.string_list(
            default = [],
            doc = """
The list of arguments for the JVM when invoking the Java compiler. Please refer to the Java
virtual machine documentation for the extensive list of possible flags for this option.
            """,
        ),
        "misc": attr.string_list(
            default = [],
            doc = """Deprecated: use javacopts instead""",
        ),
        "oneversion": attr.label(
            cfg = "exec",
            allow_files = True,
            executable = True,
            doc = """
Label of the one-version enforcement binary.
            """,
        ),
        "oneversion_whitelist": attr.label(
            allow_single_file = True,
            doc = """
Label of the one-version allowlist.
            """,
        ),
        "oneversion_allowlist_for_tests": attr.label(
            allow_single_file = True,
            doc = """
Label of the one-version allowlist for tests.
            """,
        ),
        "package_configuration": attr.label_list(
            cfg = "exec",
            providers = [JavaPackageConfigurationInfo],
            doc = """
Configuration that should be applied to the specified package groups.
            """,
        ),
        "proguard_allowlister": attr.label(
            cfg = "exec",
            executable = True,
            allow_files = True,
            default = semantics.PROGUARD_ALLOWLISTER_LABEL,
            doc = """
Label of the Proguard allowlister.
            """,
        ),
        "reduced_classpath_incompatible_processors": attr.string_list(
            doc = """Internal API, do not use!""",
        ),
        "singlejar": attr.label(
            cfg = "exec",
            allow_files = True,
            executable = True,
            doc = """
Label of the SingleJar deploy jar.
            """,
        ),
        "source_version": attr.string(
            doc = """
The Java source version (e.g., '6' or '7'). It specifies which set of code structures
are allowed in the Java source code.
            """,
        ),
        "target_version": attr.string(
            doc = """
The Java target version (e.g., '6' or '7'). It specifies for which Java runtime the class
should be build.
            """,
        ),
        "timezone_data": attr.label(
            cfg = "exec",
            allow_single_file = True,
            doc = """
Label of a resource jar containing timezone data. If set, the timezone data is added as an
implicitly runtime dependency of all java_binary rules.
            """,
        ),
        "tools": attr.label_list(
            cfg = "exec",
            allow_files = True,
            doc = """
Labels of tools available for label-expansion in jvm_opts.
            """,
        ),
        "turbine_data": attr.label_list(
            cfg = "exec",
            allow_files = True,
            doc = """
Labels of data available for label-expansion in turbine_jvm_opts.
            """,
        ),
        "turbine_jvm_opts": attr.string_list(
            doc = """
The list of arguments for the JVM when invoking turbine.
            """,
        ),
        "xlint": attr.string_list(
            default = [],
            doc = """
The list of warning to add or removes from default list. Precedes it with a dash to
removes it. Please see the Javac documentation on the -Xlint options for more information.
            """,
        ),
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
        "_legacy_any_type_attrs": attr.string_list(default = _LEGACY_ANY_TYPE_ATTRS),
    },
    fragments = ["java"],
)
