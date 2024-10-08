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
"""Configuring the C++ toolchain on Windows."""

load("@rules_cc//cc/toolchains:toolchain_config_utils.bzl", _find_vc_path = "find_vc_path", _setup_vc_env_vars = "setup_vc_env_vars")

find_vc_path = _find_vc_path
setup_vc_env_vars = _setup_vc_env_vars
