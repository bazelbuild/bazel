# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Rules for defining default_java_toolchain"""

# JVM options, without patching java.compiler and jdk.compiler modules.
BASE_JDK9_JVM_OPTS = [
    # Allow JavaBuilder to access internal javac APIs.
    "--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.model=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.processing=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.resources=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    "--add-opens=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED",
    "--add-opens=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED",
    "--add-opens=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED",
    "--add-opens=jdk.compiler/com.sun.tools.javac.parser=ALL-UNNAMED",

    # quiet warnings from com.google.protobuf.UnsafeUtil,
    # see: https://github.com/google/protobuf/issues/3781
    # and: https://github.com/bazelbuild/bazel/issues/5599
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",

    # TODO(b/64485048): Disable this option in persistent worker mode only.
    # Disable symlinks resolution cache since symlinks in exec root change
    "-Dsun.io.useCanonCaches=false",

    # Compact strings make JavaBuilder slightly slower.
    "-XX:-CompactStrings",

    # Since https://bugs.openjdk.org/browse/JDK-8153723, JVM logging goes to stdout. This
    # makes it go to stderr instead.
    "-Xlog:disable",
    "-Xlog:all=warning:stderr:uptime,level,tags",
]

JDK9_JVM_OPTS = BASE_JDK9_JVM_OPTS

DEFAULT_JAVACOPTS = [
    "-XDskipDuplicateBridges=true",
    "-XDcompilePolicy=simple",
    "-g",
    "-parameters",
    # https://github.com/bazelbuild/bazel/issues/15219
    "-Xep:ReturnValueIgnored:OFF",
    # https://github.com/bazelbuild/bazel/issues/16996
    "-Xep:IgnoredPureGetter:OFF",
    "-Xep:EmptyTopLevelDeclaration:OFF",
    "-Xep:LenientFormatStringValidation:OFF",
    "-Xep:ReturnMissingNullable:OFF",
]

# Default java_toolchain parameters
_BASE_TOOLCHAIN_CONFIGURATION = dict(
    forcibly_disable_header_compilation = False,
    genclass = ["@remote_java_tools//:GenClass"],
    header_compiler = ["@remote_java_tools//:TurbineDirect"],
    header_compiler_direct = ["@remote_java_tools//:TurbineDirect"],
    ijar = ["@bazel_tools//tools/jdk:ijar"],
    javabuilder = ["@remote_java_tools//:JavaBuilder"],
    javac_supports_workers = True,
    jacocorunner = "@remote_java_tools//:jacoco_coverage_runner_filegroup",
    jvm_opts = BASE_JDK9_JVM_OPTS,
    turbine_jvm_opts = [
        # Turbine is not a worker and parallel GC is faster for short-lived programs.
        "-XX:+UseParallelGC",
    ],
    misc = DEFAULT_JAVACOPTS,
    singlejar = ["@bazel_tools//tools/jdk:singlejar"],
    # Code to enumerate target JVM boot classpath uses host JVM. Because
    # java_runtime-s are involved, its implementation is in @bazel_tools.
    bootclasspath = ["@bazel_tools//tools/jdk:platformclasspath"],
    source_version = "8",
    target_version = "8",
    reduced_classpath_incompatible_processors = [
        "dagger.hilt.processor.internal.root.RootProcessor",  # see b/21307381
    ],
    java_runtime = "@bazel_tools//tools/jdk:remotejdk_17",
)

DEFAULT_TOOLCHAIN_CONFIGURATION = _BASE_TOOLCHAIN_CONFIGURATION

# The 'vanilla' toolchain is an unsupported alternative to the default.
#
# It does not provide any of the following features:
#   * Error Prone
#   * Strict Java Deps
#   * Reduced Classpath Optimization
#
# It uses the version of internal javac from the `--host_javabase` JDK instead
# of providing a javac. Internal javac may not be source- or bug-compatible with
# the javac that is provided with other toolchains.
#
# However it does allow using a wider range of `--host_javabase`s, including
# versions newer than the current JDK.
VANILLA_TOOLCHAIN_CONFIGURATION = dict(
    javabuilder = ["@remote_java_tools//:VanillaJavaBuilder"],
    jvm_opts = [],
    java_runtime = None,
)

# The new toolchain is using all the pre-built tools, including
# singlejar and ijar, even on remote execution. This toolchain
# should be used only when host and execution platform are the
# same, otherwise the binaries will not work on the execution
# platform.
PREBUILT_TOOLCHAIN_CONFIGURATION = dict(
    ijar = ["@bazel_tools//tools/jdk:ijar_prebuilt_binary"],
    singlejar = ["@bazel_tools//tools/jdk:prebuilt_singlejar"],
)

# The new toolchain is using all the tools from sources.
NONPREBUILT_TOOLCHAIN_CONFIGURATION = dict(
    ijar = ["@remote_java_tools//:ijar_cc_binary"],
    singlejar = ["@remote_java_tools//:singlejar_cc_bin"],
)

_DEFAULT_SOURCE_VERSION = "8"

def default_java_toolchain(name, configuration = DEFAULT_TOOLCHAIN_CONFIGURATION, toolchain_definition = True, exec_compatible_with = [], target_compatible_with = [], **kwargs):
    """Defines a remote java_toolchain with appropriate defaults for Bazel.

    Args:
        name: The name of the toolchain
        configuration: Toolchain configuration
        toolchain_definition: Whether to define toolchain target and its config setting
        exec_compatible_with: A list of constraint values that must be
            satisifed for the exec platform.
        target_compatible_with: A list of constraint values that must be
            satisifed for the target platform.
        **kwargs: More arguments for the java_toolchain target
    """

    toolchain_args = dict(_BASE_TOOLCHAIN_CONFIGURATION)
    toolchain_args.update(configuration)
    toolchain_args.update(kwargs)
    native.java_toolchain(
        name = name,
        **toolchain_args
    )
    if toolchain_definition:
        source_version = toolchain_args["source_version"]
        if source_version == _DEFAULT_SOURCE_VERSION:
            native.config_setting(
                name = name + "_default_version_setting",
                values = {"java_language_version": ""},
                visibility = ["//visibility:private"],
            )
            native.toolchain(
                name = name + "_default_definition",
                toolchain_type = "@bazel_tools//tools/jdk:toolchain_type",
                target_settings = [name + "_default_version_setting"],
                toolchain = name,
                exec_compatible_with = exec_compatible_with,
                target_compatible_with = target_compatible_with,
            )

        native.config_setting(
            name = name + "_version_setting",
            values = {"java_language_version": source_version},
            visibility = ["//visibility:private"],
        )
        native.toolchain(
            name = name + "_definition",
            toolchain_type = "@bazel_tools//tools/jdk:toolchain_type",
            target_settings = [name + "_version_setting"],
            toolchain = name,
            exec_compatible_with = exec_compatible_with,
            target_compatible_with = target_compatible_with,
        )

def java_runtime_files(name, srcs):
    """Copies the given sources out of the current Java runtime."""

    native.filegroup(
        name = name,
        srcs = srcs,
        tags = ["manual"],
    )
    for src in srcs:
        native.genrule(
            name = "gen_%s" % src,
            srcs = ["@bazel_tools//tools/jdk:current_java_runtime"],
            toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
            cmd = "cp $(JAVABASE)/%s $@" % src,
            outs = [src],
            tags = ["manual"],
        )

def _bootclasspath_impl(ctx):
    host_javabase = ctx.attr.host_javabase[java_common.JavaRuntimeInfo]

    class_dir = ctx.actions.declare_directory("%s_classes" % ctx.label.name)

    args = ctx.actions.args()
    args.add("-source")
    args.add("8")
    args.add("-target")
    args.add("8")
    args.add("-Xlint:-options")
    args.add("-d")
    args.add_all([class_dir], expand_directories = False)
    args.add(ctx.file.src)

    ctx.actions.run(
        executable = "%s/bin/javac" % host_javabase.java_home,
        mnemonic = "JavaToolchainCompileClasses",
        inputs = [ctx.file.src] + ctx.files.host_javabase,
        outputs = [class_dir],
        arguments = [args],
    )

    bootclasspath = ctx.outputs.output_jar

    inputs = [class_dir] + ctx.files.host_javabase

    args = ctx.actions.args()
    args.add("-XX:+IgnoreUnrecognizedVMOptions")
    args.add("--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED")
    args.add("--add-exports=jdk.compiler/com.sun.tools.javac.platform=ALL-UNNAMED")
    args.add("--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED")
    args.add_all("-cp", [class_dir], expand_directories = False)
    args.add("DumpPlatformClassPath")
    args.add(bootclasspath)

    system_files = ("release", "modules", "jrt-fs.jar")
    system = [f for f in ctx.files.target_javabase if f.basename in system_files]
    if len(system) != len(system_files):
        system = None
    if ctx.attr.target_javabase:
        inputs.extend(ctx.files.target_javabase)
        args.add(ctx.attr.target_javabase[java_common.JavaRuntimeInfo].java_home)

    ctx.actions.run(
        executable = str(host_javabase.java_executable_exec_path),
        mnemonic = "JavaToolchainCompileBootClasspath",
        inputs = inputs,
        outputs = [bootclasspath],
        arguments = [args],
    )
    return [
        DefaultInfo(files = depset([bootclasspath])),
        java_common.BootClassPathInfo(
            bootclasspath = [bootclasspath],
            system = system,
        ),
        OutputGroupInfo(jar = [bootclasspath]),
    ]

_bootclasspath = rule(
    implementation = _bootclasspath_impl,
    attrs = {
        "host_javabase": attr.label(
            cfg = "exec",
            providers = [java_common.JavaRuntimeInfo],
        ),
        "output_jar": attr.output(mandatory = True),
        "src": attr.label(
            cfg = "exec",
            allow_single_file = True,
        ),
        "target_javabase": attr.label(
            providers = [java_common.JavaRuntimeInfo],
        ),
    },
)

def bootclasspath(name, **kwargs):
    _bootclasspath(
        name = name,
        output_jar = name + ".jar",
        **kwargs
    )
