# Copyright 2017 The Bazel Authors. All rights reserved.
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

"""Bazel rules for creating Java toolchains."""

load("@rules_java//java:defs.bzl", "java_toolchain")

JDK8_JVM_OPTS = [
    "-Xbootclasspath/p:$(location @bazel_tools//tools/jdk:javac_jar)",
]

JDK9_JVM_OPTS = [
    # Allow JavaBuilder to access internal javac APIs.
    "--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    "--add-opens=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED",

    # override the javac in the JDK.
    "--patch-module=java.compiler=$(location @bazel_tools//tools/jdk:java_compiler_jar)",
    "--patch-module=jdk.compiler=$(location @bazel_tools//tools/jdk:jdk_compiler_jar)",

    # quiet warnings from com.google.protobuf.UnsafeUtil,
    # see: https://github.com/google/protobuf/issues/3781
    # and: https://github.com/bazelbuild/bazel/issues/5599
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
]

DEFAULT_JAVACOPTS = [
    "-XDskipDuplicateBridges=true",
    "-XDcompilePolicy=simple",
    "-g",
    "-parameters",
]

DEFAULT_TOOLCHAIN_CONFIGURATION = {
    "forcibly_disable_header_compilation": 0,
    "genclass": ["@bazel_tools//tools/jdk:genclass"],
    "header_compiler": ["@bazel_tools//tools/jdk:turbine_direct"],
    "header_compiler_direct": ["@bazel_tools//tools/jdk:turbine_direct"],
    "ijar": ["@bazel_tools//tools/jdk:ijar"],
    "javabuilder": ["@bazel_tools//tools/jdk:javabuilder"],
    "jacocorunner": "@bazel_tools//tools/jdk:JacocoCoverageFilegroup",
    "tools": [
        "@bazel_tools//tools/jdk:javac_jar",
        "@bazel_tools//tools/jdk:java_compiler_jar",
        "@bazel_tools//tools/jdk:jdk_compiler_jar",
    ],
    "javac_supports_workers": 1,
    "jvm_opts": select({
        "@bazel_tools//src/conditions:openbsd": JDK8_JVM_OPTS,
        "//conditions:default": JDK9_JVM_OPTS,
    }),
    "misc": DEFAULT_JAVACOPTS,
    "singlejar": ["@bazel_tools//tools/jdk:singlejar"],
    "bootclasspath": ["@bazel_tools//tools/jdk:platformclasspath"],
    "source_version": "8",
    "target_version": "8",
}

def default_java_toolchain(name, **kwargs):
    """Defines a remote java_toolchain with appropriate defaults for Bazel."""

    toolchain_args = dict(DEFAULT_TOOLCHAIN_CONFIGURATION)
    toolchain_args.update(kwargs)
    java_toolchain(
        name = name,
        **toolchain_args
    )

def java_runtime_files(name, srcs):
    """Copies the given sources out of the current Java runtime."""

    native.filegroup(
        name = name,
        srcs = srcs,
    )
    for src in srcs:
        native.genrule(
            name = "gen_%s" % src,
            srcs = ["@bazel_tools//tools/jdk:current_java_runtime"],
            toolchains = ["@bazel_tools//tools/jdk:current_java_runtime"],
            cmd = "cp $(JAVABASE)/%s $@" % src,
            outs = [src],
        )

def _bootclasspath_impl(ctx):
    host_javabase = ctx.attr.host_javabase[java_common.JavaRuntimeInfo]

    # explicitly list output files instead of using TreeArtifact to work around
    # https://github.com/bazelbuild/bazel/issues/6203
    classes = [
        "DumpPlatformClassPath.class",
    ]

    class_outputs = [
        ctx.actions.declare_file("%s_classes/%s" % (ctx.label.name, clazz))
        for clazz in classes
    ]

    args = ctx.actions.args()
    args.add("-source")
    args.add("8")
    args.add("-target")
    args.add("8")
    args.add("-Xlint:-options")
    args.add("-cp")
    args.add("%s/lib/tools.jar" % host_javabase.java_home)
    args.add("-d")
    args.add(class_outputs[0].dirname)
    args.add(ctx.file.src)

    ctx.actions.run(
        executable = "%s/bin/javac" % host_javabase.java_home,
        inputs = [ctx.file.src] + ctx.files.host_javabase,
        outputs = class_outputs,
        arguments = [args],
    )

    bootclasspath = ctx.outputs.output_jar

    inputs = class_outputs + ctx.files.host_javabase

    args = ctx.actions.args()
    args.add("-XX:+IgnoreUnrecognizedVMOptions")
    args.add("--add-exports=jdk.compiler/com.sun.tools.javac.platform=ALL-UNNAMED")
    args.add_joined(
        "-cp",
        [class_outputs[0].dirname, "%s/lib/tools.jar" % host_javabase.java_home],
        join_with = ctx.configuration.host_path_separator,
    )
    args.add("DumpPlatformClassPath")
    args.add(bootclasspath)

    if ctx.attr.target_javabase:
        inputs.extend(ctx.files.target_javabase)
        args.add(ctx.attr.target_javabase[java_common.JavaRuntimeInfo].java_home)

    ctx.actions.run(
        executable = str(host_javabase.java_executable_exec_path),
        inputs = inputs,
        outputs = [bootclasspath],
        arguments = [args],
    )
    return [
        DefaultInfo(files = depset([bootclasspath])),
        OutputGroupInfo(jar = [bootclasspath]),
    ]

_bootclasspath = rule(
    implementation = _bootclasspath_impl,
    attrs = {
        "host_javabase": attr.label(
            cfg = "host",
            providers = [java_common.JavaRuntimeInfo],
        ),
        "src": attr.label(
            cfg = "host",
            allow_single_file = True,
        ),
        "target_javabase": attr.label(
            providers = [java_common.JavaRuntimeInfo],
        ),
        "output_jar": attr.output(mandatory = True),
    },
)

def bootclasspath(name, **kwargs):
    _bootclasspath(
        name = name,
        output_jar = name + ".jar",
        **kwargs
    )
