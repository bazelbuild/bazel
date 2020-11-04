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
    genclass = [Label("//:GenClass")],
    header_compiler = [Label("//:TurbineDirect")],
    header_compiler_direct = [Label("//:TurbineDirect")],
    ijar = [Label("//:ijar")],
    javabuilder = [Label("//:JavaBuilder")],
    javac_supports_workers = True,
    jacocorunner = Label("//:jacoco_coverage_runner_filegroup"),
    jvm_opts = JDK9_JVM_OPTS,
    misc = _DEFAULT_JAVACOPTS,
    singlejar = [Label("//:singlejar")],
    # Code to enumerate target JVM boot classpath uses host JVM. Because
    # java_runtime-s are involved, its implementation is in @bazel_tools.
    bootclasspath = ["@bazel_tools//tools/jdk:platformclasspath"],
    source_version = "8",
    target_version = "8",
)

_LABEL_LISTS = [
    "bootclasspath",
    "extclasspath",
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
    if k in _LABELS and type(v) == type(Label("//a")):
        return Label(v)
    if k in _LABEL_LISTS and type(v) == type([Label("//a")]):
        return [Label(label) for label in v]
    return v

def java_toolchain_default(name, **kwargs):
    """Defines a java_toolchain with appropriate defaults for Bazel."""

    toolchain_args = dict(_BASE_TOOLCHAIN_CONFIGURATION)
    toolchain_args.update({k: _to_label(k, v) for k, v in kwargs.items()})
    native.java_toolchain(
        name = name,
        **toolchain_args
    )
