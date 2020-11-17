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

_DEFAULT_JAVACOPTS = [
    "-XDskipDuplicateBridges=true",
    "-XDcompilePolicy=simple",
    "-g",
    "-parameters",
]

# JVM options, without patching java.compiler and jdk.compiler modules.
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

    # quiet warnings from com.google.protobuf.UnsafeUtil,
    # see: https://github.com/google/protobuf/issues/3781
    # and: https://github.com/bazelbuild/bazel/issues/5599
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
]

# java_toolchain parameters without specifying javac, java.compiler,
# jdk.compiler module, and jvm_opts
_BASE_TOOLCHAIN_CONFIGURATION = dict(
    forcibly_disable_header_compilation = False,
    genclass = ["//:GenClass"],
    header_compiler = ["//:TurbineDirect"],
    header_compiler_direct = ["//:TurbineDirect"],
    ijar = ["//:ijar"],
    javabuilder = ["//:JavaBuilder"],
    javac_supports_workers = True,
    jacocorunner = "//:jacoco_coverage_runner_filegroup",
    jvm_opts = JDK9_JVM_OPTS,
    misc = _DEFAULT_JAVACOPTS,
    singlejar = ["//:singlejar"],
    # Code to enumerate target JVM boot classpath uses host JVM. Because
    # java_runtime-s are involved, its implementation is in @bazel_tools.
    bootclasspath = ["@bazel_tools//tools/jdk:platformclasspath"],
    source_version = "8",
    target_version = "8",
)

JVM8_TOOLCHAIN_CONFIGURATION = dict(
    tools = ["//:javac_jar"],
    jvm_opts = ["-Xbootclasspath/p:$(location //:javac_jar)"],
)

JAVABUILDER_TOOLCHAIN_CONFIGURATION = dict(
    jvm_opts = [
        # In JDK9 we have seen a ~30% slow down in JavaBuilder performance when using
        # G1 collector and having compact strings enabled.
        "-XX:+UseParallelOldGC",
        "-XX:-CompactStrings",
        # override the javac in the JDK.
        "--patch-module=java.compiler=$(location //:java_compiler_jar)",
        "--patch-module=jdk.compiler=$(location //:jdk_compiler_jar)",
    ] + JDK9_JVM_OPTS,
    tools = [
        "//:java_compiler_jar",
        "//:jdk_compiler_jar",
    ],
)

# The 'vanilla' toolchain is an unsupported alternative to the default.
#
# It does not provide any of the following features:
#   * Error Prone
#   * Strict Java Deps
#   * Header Compilation
#   * Reduced Classpath Optimization
#
# It uses the version of internal javac from the `--host_javabase` JDK instead
# of providing a javac. Internal javac may not be source- or bug-compatible with
# the javac that is provided with other toolchains.
#
# However it does allow using a wider range of `--host_javabase`s, including
# versions newer than the current JDK.
VANILLA_TOOLCHAIN_CONFIGURATION = dict(
    forcibly_disable_header_compilation = True,
    javabuilder = ["//:VanillaJavaBuilder"],
    jvm_opts = [],
)

# The new toolchain is using all the pre-built tools, including
# singlejar and ijar, even on remote execution. This toolchain
# should be used only when host and execution platform are the
# same, otherwise the binaries will not work on the execution
# platform.
PREBUILT_TOOLCHAIN_CONFIGURATION = dict(
    jvm_opts = [
        # In JDK9 we have seen a ~30% slow down in JavaBuilder performance when using
        # G1 collector and having compact strings enabled.
        "-XX:+UseParallelOldGC",
        "-XX:-CompactStrings",
        # override the javac in the JDK.
        "--patch-module=java.compiler=$(location //:java_compiler_jar)",
        "--patch-module=jdk.compiler=$(location //:jdk_compiler_jar)",
    ] + JDK9_JVM_OPTS,
    tools = [
        "//:java_compiler_jar",
        "//:jdk_compiler_jar",
    ],
    ijar = ["//:ijar_prebuilt_binary"],
    singlejar = ["//:prebuilt_singlejar"],
)

_LABEL_LISTS = [
    "bootclasspath",
    "javac",
    "tools",
    "javabuilder",
    "singlejar",
    "genclass",
    "resourcejar",
    "ijar",
    "header_compiler",
    "header_compiler_direct",
    "package_configuration",
]

_LABELS = [
    "timezone_data",
    "oneversion",
    "oneversion_whitelist",
    "jacocorunner",
    "proguard_allowlister",
    "java_runtime",
]

# Converts values to labels, so that they are resolved relative to this java_tools repository
def _to_label(k, v):
    if k in _LABELS and type(v) != type(Label("//a")):
        return Label(v)
    if k in _LABEL_LISTS and type(v) == type([Label("//a")]):
        return [Label(label) if type(label) == type("") else label for label in v]
    return v

# Makes labels in jvm_opts absolute, that is replaces " //:"  with " @repo//:".
def _format_jvm_opts(toolchain_args, repo):
    jvm_opts = toolchain_args["jvm_opts"]
    if [opt for opt in jvm_opts if opt.find(" :") >= 0] != []:
        fail("Relative labels are not supported in jvm_opts parameter.")
    jvm_opts = [opt.replace(" //:", " @{repo}//:").format(repo = repo) for opt in jvm_opts]
    return dict(toolchain_args, jvm_opts = jvm_opts)

def java_toolchain_default(name, configuration = dict(), **kwargs):
    """Defines a java_toolchain with appropriate defaults for Bazel."""

    toolchain_args = dict(_BASE_TOOLCHAIN_CONFIGURATION)
    toolchain_args.update(configuration)
    toolchain_args.update(kwargs)
    toolchain_args = {k: _to_label(k, v) for k, v in toolchain_args.items()}
    toolchain_args = _format_jvm_opts(toolchain_args, Label("//x").workspace_name)
    native.java_toolchain(
        name = name,
        **toolchain_args
    )
