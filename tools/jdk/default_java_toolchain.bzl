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

JVM_OPTS = [
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
    "ijar": ["@bazel_tools//tools/jdk:ijar"],
    "javabuilder": ["@bazel_tools//tools/jdk:javabuilder"],
    "javac": ["@bazel_tools//third_party/java/jdk/langtools:javac_jar"],
    "tools": [
        "@bazel_tools//third_party/java/jdk/langtools:java_compiler_jar",
        "@bazel_tools//third_party/java/jdk/langtools:jdk_compiler_jar",
    ],
    "javac_supports_workers": 1,
    "jvm_opts": JVM_OPTS,
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
            srcs = ["//tools/jdk:current_java_runtime"],
            toolchains = ["//tools/jdk:current_java_runtime"],
            cmd = "cp $(JAVABASE)/%s $@" % src,
            outs = [src],
        )
