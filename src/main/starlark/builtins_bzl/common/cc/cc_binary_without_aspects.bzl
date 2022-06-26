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

"""Exports cc_binary variant without aspects.

If dynamic_deps attribute is not specified we do not propagate
aspects.
"""

load(":common/cc/cc_binary.bzl", "make_cc_binary")
load(":common/cc/cc_binary_attrs.bzl", "cc_binary_attrs_without_aspects")

cc_binary = make_cc_binary(cc_binary_attrs_without_aspects)
