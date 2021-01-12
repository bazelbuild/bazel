# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Exported builtins symbols that are not specific to OSS Bazel."""

exported_toplevels = {
    # This dummy symbol is not part of the public API; it is only used to test
    # that builtins injection is working properly. Its built-in value is
    # "original value".
    "_builtins_dummy": "overridden value",
}
exported_rules = {}
exported_to_java = {}
