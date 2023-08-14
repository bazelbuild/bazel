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

"""Exports cc_toolchain with target attributes"""

load(":common/cc/cc_toolchain.bzl", "make_cc_toolchain")
load(":common/cc/cc_toolchain_attrs.bzl", "apple_cc_toolchain_attrs_target", "cc_toolchain_attrs_target")

cc_toolchain = make_cc_toolchain(cc_toolchain_attrs_target)
apple_cc_toolchain = make_cc_toolchain(apple_cc_toolchain_attrs_target)
