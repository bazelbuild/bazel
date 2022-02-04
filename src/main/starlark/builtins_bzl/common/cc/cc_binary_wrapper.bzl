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

"""Macro encapsulating cc_binary rule implementation.

This is to avoid propagating aspect on certain attributes in case
dynamic_deps attribute is not specified.
"""

load(":common/cc/cc_binary_with_aspects.bzl", cc_binary_with_aspects = "cc_binary")
load(":common/cc/cc_binary_without_aspects.bzl", cc_binary_without_aspects = "cc_binary")

def cc_binary(**kwargs):
    # Propagate an aspect if dynamic_deps attribute is specified.
    if "dynamic_deps" in kwargs and len(kwargs["dynamic_deps"]) > 0:
        cc_binary_with_aspects(**kwargs)
    else:
        cc_binary_without_aspects(**kwargs)
