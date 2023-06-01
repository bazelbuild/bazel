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

load("@rules_java//toolchains:local_java_repository.bzl", _local_java_repository = "local_java_repository", _local_java_runtime = "local_java_runtime")

def local_java_repository(name, **kwargs):
    _local_java_repository(name, **kwargs)
    native.register_toolchains("@" + name + "//:runtime_toolchain_definition")

local_java_runtime = _local_java_runtime
