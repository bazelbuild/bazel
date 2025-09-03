# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""
A wrapper around java_binary that forces it to be built with `-c opt`.

This is useful for benchmark targets.
"""

load("@rules_java//java:java_binary.bzl", "java_binary")
load("@with_cfg.bzl", "with_cfg")

java_opt_binary, _java_opt_binary = with_cfg(java_binary).set("compilation_mode", "opt").build()
