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

"""Rules to create windows configurations"""

def create_windows_host_config(name = "windows_host_config"):
    native.config_setting(
        name = "host_windows_x64_constraint",
        values = {"host_cpu": "x64_windows"},
    )

    native.config_setting(
        name = "host_windows_arm64_constraint",
        values = {"host_cpu": "arm64_windows"},
    )

    conditions = select({
        "host_windows_arm64_constraint": "host_windows_arm64_constraint",
        "//conditions:default": "host_windows_x64_constraint",
    })

    native.alias(name = "host_windows", actual = conditions, visibility = ["//visibility:public"])
