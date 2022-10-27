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
"""Macro to wrap the py_runtime rule."""

load(":common/python/py_runtime_rule.bzl", py_runtime_rule = "py_runtime")

# NOTE: The function name is purposefully selected to match the underlying
# rule name so that e.g. 'generator_function' shows as the same name so
# that it is less confusing to users.
def py_runtime(**kwargs):
    py_runtime_rule(**kwargs)
