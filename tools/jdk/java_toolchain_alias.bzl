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

"""Redirect symbols from rules_java to keep backward-compatibility."""

load("@rules_java//toolchains:java_toolchain_alias.bzl", _java_host_runtime_alias = "java_host_runtime_alias", _java_runtime_alias = "java_runtime_alias", _java_toolchain_alias = "java_toolchain_alias")

java_toolchain_alias = _java_toolchain_alias
java_runtime_alias = _java_runtime_alias
java_host_runtime_alias = _java_host_runtime_alias
