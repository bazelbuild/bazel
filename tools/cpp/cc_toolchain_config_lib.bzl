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

load(
    "@rules_cc//cc:cc_toolchain_config_lib.bzl",
    _action_config = "action_config",
    _artifact_name_pattern = "artifact_name_pattern",
    _env_entry = "env_entry",
    _env_set = "env_set",
    _feature = "feature",
    _feature_set = "feature_set",
    _flag_group = "flag_group",
    _flag_set = "flag_set",
    _make_variable = "make_variable",
    _tool = "tool",
    _tool_path = "tool_path",
    _with_feature_set = "with_feature_set",
)

action_config = _action_config
artifact_name_pattern = _artifact_name_pattern
env_entry = _env_entry
env_set = _env_set
feature = _feature
feature_set = _feature_set
flag_group = _flag_group
flag_set = _flag_set
make_variable = _make_variable
tool = _tool
tool_path = _tool_path
with_feature_set = _with_feature_set
