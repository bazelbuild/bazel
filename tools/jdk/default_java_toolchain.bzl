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

JDK8_JVM_OPTS = [
    "-Xbootclasspath/p:$(location @bazel_tools//third_party/java/jdk/langtools:javac_jar)",
]

JDK9_JVM_OPTS = [
    # In JDK9 we have seen a ~30% slow down in JavaBuilder performance when using
    # G1 collector and having compact strings enabled.
    "-XX:+UseParallelOldGC",
    "-XX:-CompactStrings",
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
    "--patch-module=java.compiler=$(location @bazel_tools//third_party/java/jdk/langtools:java_compiler_jar)",
    "--patch-module=jdk.compiler=$(location @bazel_tools//third_party/java/jdk/langtools:jdk_compiler_jar)",

    # quiet warnings from com.google.protobuf.UnsafeUtil,
    # see: https://github.com/google/protobuf/issues/3781
    # and: https://github.com/bazelbuild/bazel/issues/5599
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
]

DEFAULT_JAVACOPTS = [
    "-XDskipDuplicateBridges=true",
    "-g",
    "-parameters",
]

PROTO_JAVACOPTS = [
    # Restrict protos to Java 7 so that they are compatible with Android.
    "-source",
    "7",
    "-target",
    "7",
]

COMPATIBLE_JAVACOPTS = {
    "proto": PROTO_JAVACOPTS,
}

DEFAULT_TOOLCHAIN_CONFIGURATION = {
    "forcibly_disable_header_compilation": 0,
    "genclass": ["@bazel_tools//tools/jdk:genclass"],
    "header_compiler": ["@bazel_tools//tools/jdk:turbine"],
    "header_compiler_direct": ["@bazel_tools//tools/jdk:turbine_direct"],
    "ijar": ["@bazel_tools//tools/jdk:ijar"],
    "javabuilder": ["@bazel_tools//tools/jdk:javabuilder"],
    "javac": ["@bazel_tools//third_party/java/jdk/langtools:javac_jar"],
    "tools": [
        "@bazel_tools//third_party/java/jdk/langtools:java_compiler_jar",
        "@bazel_tools//third_party/java/jdk/langtools:jdk_compiler_jar",
    ],
    "javac_supports_workers": 1,
    "jvm_opts": JDK9_JVM_OPTS,
    "misc": DEFAULT_JAVACOPTS,
    "compatible_javacopts": COMPATIBLE_JAVACOPTS,
    "singlejar": ["@bazel_tools//tools/jdk:singlejar"],
}

def default_java_toolchain(name, **kwargs):
    """Defines a java_toolchain with appropriate defaults for Bazel."""

    toolchain_args = dict(DEFAULT_TOOLCHAIN_CONFIGURATION)
    toolchain_args.update(kwargs)

    native.java_toolchain(
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

def _bootclasspath(ctx):
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

    bootclasspath = ctx.outputs.jar

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

bootclasspath = rule(
    implementation = _bootclasspath,
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
    },
    outputs = {"jar": "%{name}.jar"},
)
