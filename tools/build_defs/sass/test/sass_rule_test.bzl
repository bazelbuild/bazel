# Copyright 2015 The Bazel Authors. All rights reserved.
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

load(
    "//tools/build_defs/sass:sass.bzl",
    "sass_binary",
)

load(
    "//tools:build_rules/test_rules.bzl",
    "success_target",
    "successful_test",
    "failure_target",
    "failed_test",
    "assert_",
    "strip_prefix",
    "expectation_description",
    "check_results",
    "load_results",
    "analysis_results",
    "rule_test",
    "file_test",
)

def _sass_binary_test(package):
    rule_test(
        name = "hello_world_rule_test",
        generates = ["hello_world.css", "hello_world.css.map"],
        rule = package + "/hello_world:hello_world",
    )

def sass_rule_test(package):
    """Issue simple tests on sass rules."""
    _sass_binary_test(package)
