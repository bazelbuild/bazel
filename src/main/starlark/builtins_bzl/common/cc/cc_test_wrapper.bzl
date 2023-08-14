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

"""cc_test Starlark implementation."""

load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/cc_test_no_linkstatic.bzl", _cc_test_no_linkstatic = "cc_test")
load(":common/cc/cc_test_with_linkstatic.bzl", _cc_test_with_linkstatic = "cc_test")
load(":common/cc/cc_test_no_linkstatic_aspects.bzl", _cc_test_no_linkstatic_aspects = "cc_test")
load(":common/cc/cc_test_with_linkstatic_aspects.bzl", _cc_test_with_linkstatic_aspects = "cc_test")

def cc_test(**kwargs):
    """Entry point for cc_test rules.

    This avoids propagating aspects on certain attributes if dynamic_deps attribute is unset.

    It also serves to detect if the `linkstatic` attribute was explicitly set or not.
    This is to workaround a deficiency in Starlark attributes.
    (See: https://github.com/bazelbuild/bazel/issues/14434)

    Args:
        **kwargs: Arguments suitable for cc_test.
    """

    # Propagate an aspect if dynamic_deps attribute is specified.
    if "dynamic_deps" in kwargs and cc_helper.is_non_empty_list_or_select(kwargs["dynamic_deps"], "dynamic_deps"):
        if "linkstatic" in kwargs:
            _cc_test_with_linkstatic_aspects(**kwargs)
        else:
            _cc_test_no_linkstatic_aspects(**kwargs)
    elif "linkstatic" in kwargs:
        _cc_test_with_linkstatic(**kwargs)
    else:
        _cc_test_no_linkstatic(**kwargs)
