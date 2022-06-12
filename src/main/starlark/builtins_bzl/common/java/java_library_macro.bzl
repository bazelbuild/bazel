# Copyright 2021 The Bazel Authors. All rights reserved.
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
Definition of java_library macro (handles implicit outputs).
"""

load(":common/java/java_semantics.bzl", "semantics")
load(":common/java/java_library.bzl", _java_library = "java_library")

filegroup = _builtins.toplevel.native.filegroup

def java_library(name, **kwargs):
    semantics.macro_preprocess(kwargs)

    _java_library(name = name, **kwargs)
