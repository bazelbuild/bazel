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

"""Macro encapsulating cc_toolchain and apple_cc_toolchain rule implementation.

This is to mark certain attributes with "target" configuration.
"""

load(":common/cc/cc_toolchain_exec.bzl", apple_cc_toolchain_exec = "apple_cc_toolchain", cc_toolchain_exec = "cc_toolchain")
load(":common/cc/cc_toolchain_target.bzl", apple_cc_toolchain_target = "apple_cc_toolchain", cc_toolchain_target = "cc_toolchain")

def cc_toolchain(**kwargs):
    if "exec_transition_for_inputs" in kwargs and not kwargs["exec_transition_for_inputs"]:
        cc_toolchain_target(**kwargs)
    else:
        cc_toolchain_exec(**kwargs)

def apple_cc_toolchain(**kwargs):
    if "exec_transition_for_inputs" in kwargs and not kwargs["exec_transition_for_inputs"]:
        apple_cc_toolchain_target(**kwargs)
    else:
        apple_cc_toolchain_exec(**kwargs)
