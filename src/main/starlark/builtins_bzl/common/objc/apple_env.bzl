# Copyright 2024 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License(**kwargs): Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing(**kwargs): software
# distributed under the License is distributed on an "AS IS" BASIS(**kwargs):
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND(**kwargs): either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to retrieve the environment to set for Apple actions."""

def apple_host_system_env(xcode_version_info):
    if not xcode_version_info:
        return {}
    return {"XCODE_VERSION_OVERRIDE": str(xcode_version_info.xcode_version())}

def target_apple_env(_xcode_version_info, platform):
    return {
        "APPLE_SDK_PLATFORM": platform.name_in_plist,
    }
