# Copyright 2020 The Bazel Authors. All rights reserved.
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
    "-Xbootclasspath/p:$(location :javac_jar)",
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
    "--patch-module=java.compiler=$(location :java_compiler_jar)",
    "--patch-module=jdk.compiler=$(location :jdk_compiler_jar)",

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

DEFAULT_TOOLCHAIN_CONFIGURATION = {
    "forcibly_disable_header_compilation": 0,
    "genclass": [":GenClass"],
    "header_compiler": [":Turbine"],
    "header_compiler_direct": [":TurbineDirect"],
    "ijar": [":ijar"],
    "jacocorunner": ":jacoco_coverage_runner_filegroup",
    "javabuilder": [":JavaBuilder"],
    "javac": [":javac_jar"],
    "tools": [
        ":java_compiler_jar",
        ":jdk_compiler_jar",
    ],
    "javac_supports_workers": 1,
    "jvm_opts": JDK9_JVM_OPTS,
    "misc": DEFAULT_JAVACOPTS,
    "singlejar": [":singlejar"],
    "bootclasspath": ["@bazel_tools//tools/jdk:platformclasspath"],
    "source_version": "8",
    "target_version": "8",
}

def java_toolchain_default(name, **kwargs):
    """Defines a java_toolchain with appropriate defaults for Bazel."""

    toolchain_args = dict(DEFAULT_TOOLCHAIN_CONFIGURATION)
    toolchain_args.update(kwargs)
    native.java_toolchain(
        name = name,
        **toolchain_args
    )

JDK_NOJAVAC_JVM_OPTS = [
    # Allow JavaBuilder to access internal javac APIs.
    "--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
    "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
    "--add-opens=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED",

    # quiet warnings from com.google.protobuf.UnsafeUtil,
    # see: https://github.com/google/protobuf/issues/3781
    # and: https://github.com/bazelbuild/bazel/issues/5599
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
]

NOJAVAC_TOOLCHAIN_CONFIGURATION = {
    "forcibly_disable_header_compilation": 0,
    "genclass": [":GenClass"],
    "header_compiler": [":Turbine"],
    "header_compiler_direct": [":TurbineDirect"],
    "ijar": [":ijar"],
    "javabuilder": [":JavaBuilder"],
    "javac_supports_workers": 1,
    "jvm_opts": JDK_NOJAVAC_JVM_OPTS,
    "misc": DEFAULT_JAVACOPTS,
    "singlejar": [":singlejar"],
    "bootclasspath": ["@bazel_tools//tools/jdk:platformclasspath"],
    "source_version": "8",
    "target_version": "8",
}

def java_toolchain_nojavac(name, **kwargs):
    """Defines a java_toolchain without overriding javac for Bazel."""

    toolchain_args = dict(NOJAVAC_TOOLCHAIN_CONFIGURATION)
    toolchain_args.update(kwargs)
    native.java_toolchain(
        name = name,
        **toolchain_args
    )
