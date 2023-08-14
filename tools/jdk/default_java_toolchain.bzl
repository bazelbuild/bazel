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

"""Redirect symbols from rules_java to keep backward-compatibility."""

load(
    "@rules_java//toolchains:default_java_toolchain.bzl",
    _BASE_JDK9_JVM_OPTS = "BASE_JDK9_JVM_OPTS",
    _DEFAULT_JAVACOPTS = "DEFAULT_JAVACOPTS",
    _DEFAULT_TOOLCHAIN_CONFIGURATION = "DEFAULT_TOOLCHAIN_CONFIGURATION",
    _JDK9_JVM_OPTS = "JDK9_JVM_OPTS",
    _NONPREBUILT_TOOLCHAIN_CONFIGURATION = "NONPREBUILT_TOOLCHAIN_CONFIGURATION",
    _PREBUILT_TOOLCHAIN_CONFIGURATION = "PREBUILT_TOOLCHAIN_CONFIGURATION",
    _VANILLA_TOOLCHAIN_CONFIGURATION = "VANILLA_TOOLCHAIN_CONFIGURATION",
    _bootclasspath = "bootclasspath",
    _default_java_toolchain = "default_java_toolchain",
    _java_runtime_files = "java_runtime_files",
)

default_java_toolchain = _default_java_toolchain
java_runtime_files = _java_runtime_files
bootclasspath = _bootclasspath
DEFAULT_TOOLCHAIN_CONFIGURATION = _DEFAULT_TOOLCHAIN_CONFIGURATION
BASE_JDK9_JVM_OPTS = _BASE_JDK9_JVM_OPTS
JDK9_JVM_OPTS = _JDK9_JVM_OPTS
DEFAULT_JAVACOPTS = _DEFAULT_JAVACOPTS
VANILLA_TOOLCHAIN_CONFIGURATION = _VANILLA_TOOLCHAIN_CONFIGURATION
PREBUILT_TOOLCHAIN_CONFIGURATION = _PREBUILT_TOOLCHAIN_CONFIGURATION
NONPREBUILT_TOOLCHAIN_CONFIGURATION = _NONPREBUILT_TOOLCHAIN_CONFIGURATION
