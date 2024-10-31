# Copyright 2024 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuring the C++ toolchain on Unix."""

load("@rules_cc//cc/private/toolchain:unix_cc_toolchain_config.bzl", _cc_toolchain_config = "cc_toolchain_config")

# DEPRECATED: This symbol is provided for backwards compatibility only and
# should not be used in new code.
cc_toolchain_config = _cc_toolchain_config
