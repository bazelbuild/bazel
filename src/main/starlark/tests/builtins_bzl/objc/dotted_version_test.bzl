# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""Unit tests for apple_common.dotted_version()"""

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")

tested_dotted_version = apple_common.dotted_version

def _assert_v1_less_v2(env, v1, v2):
    dv1 = tested_dotted_version(v1)
    dv2 = tested_dotted_version(v2)
    asserts.true(env, dv1.compare_to(dv2) < 0)
    asserts.true(env, dv2.compare_to(dv1) > 0)
    asserts.equals(env, dv1.compare_to(dv1), 0)

def _assert_v1_equal_v2(env, v1, v2):
    dv1 = tested_dotted_version(v1)
    dv2 = tested_dotted_version(v2)
    asserts.equals(env, dv1.compare_to(dv2), 0)
    asserts.equals(env, dv2.compare_to(dv1), 0)

def _compare_equal_length_versions_impl(ctx):
    env = unittest.begin(ctx)
    v1 = "5"
    v2 = "6"
    _assert_v1_less_v2(env, v1, v2)
    v3 = "3.4"
    v4 = "3.5"
    _assert_v1_less_v2(env, v3, v4)
    v5 = "1.2.3"
    v6 = "1.2.4"
    _assert_v1_less_v2(env, v5, v6)
    v7 = "1.2.5"
    v8 = "1.3.4"
    _assert_v1_less_v2(env, v7, v8)
    v9 = "1.8"
    v10 = "1.12"  # make sure component's first_number is compared as int, not as string
    _assert_v1_less_v2(env, v9, v10)
    return unittest.end(env)

compare_equal_length_versions_test = unittest.make(_compare_equal_length_versions_impl)

def _compare_different_length_versions_impl(ctx):
    env = unittest.begin(ctx)
    v1 = "9"
    v2 = "9.7.4"
    _assert_v1_less_v2(env, v1, v2)
    v3 = "2.1"
    v4 = "2.1.8"
    _assert_v1_less_v2(env, v3, v4)
    return unittest.end(env)

compare_different_length_versions_test = unittest.make(_compare_different_length_versions_impl)

def _compare_versions_with_alphanum_components_impl(ctx):
    env = unittest.begin(ctx)
    v1 = "1.5alpha"
    _assert_v1_equal_v2(env, v1, v1)
    v3 = "1.5alpha"
    v4 = "1.5beta"
    _assert_v1_less_v2(env, v3, v4)
    v5 = "1.5beta2"
    v6 = "1.5beta3"
    _assert_v1_less_v2(env, v5, v6)
    v7 = "1.5gamma5"
    v8 = "1.5gamma29"  # make sure component's second_number is compared as int, not as string
    _assert_v1_less_v2(env, v7, v8)
    v9 = "1.5alpha9"
    v10 = "1.5beta7"
    _assert_v1_less_v2(env, v9, v10)
    return unittest.end(env)

compare_versions_with_alphanum_components_test = unittest.make(_compare_versions_with_alphanum_components_impl)

def _check_description_is_ignored_impl(ctx):
    env = unittest.begin(ctx)
    v1 = "1.5.decription"
    v2 = "1.5"
    _assert_v1_equal_v2(env, v1, v2)
    v3 = "1.5.decription.6.7"  # everything after the description is ignored
    v4 = "1.5"
    _assert_v1_equal_v2(env, v3, v4)
    env = unittest.begin(ctx)
    v5 = "9.description"
    v6 = "9.7.4"
    _assert_v1_less_v2(env, v5, v6)
    return unittest.end(env)

check_description_is_ignored_test = unittest.make(_check_description_is_ignored_impl)

def dotted_version_test_suite(name):
    unittest.suite(
        name,
        compare_equal_length_versions_test,
        compare_different_length_versions_test,
        compare_versions_with_alphanum_components_test,
        check_description_is_ignored_test,
    )
