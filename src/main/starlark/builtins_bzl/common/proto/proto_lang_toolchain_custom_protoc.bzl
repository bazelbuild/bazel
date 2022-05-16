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

"""Defines a proto_lang_toolchain rule class with custom proto compiler.

There are two physical rule classes for proto_lang_toolchain and we want both of them
to have a name string of "proto_lang_toolchain".
"""

load(":common/proto/proto_lang_toolchain.bzl", "make_proto_lang_toolchain")

proto_lang_toolchain = make_proto_lang_toolchain(custom_proto_compiler = True)
