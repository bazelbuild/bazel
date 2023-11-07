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

load(":common/cc/cc_test.bzl", _cc_test = "cc_test")
load(":common/cc/semantics.bzl", "semantics")

def cc_test(**kwargs):
    """Entry point for cc_test rules.

    It  serves to detect if the `linkstatic` attribute was explicitly set or not.
    This is to workaround a deficiency in Starlark attributes.
    (See: https://github.com/bazelbuild/bazel/issues/14434)

    Args:
        **kwargs: Arguments suitable for cc_test.
    """

    if "linkstatic" not in kwargs:
        kwargs["linkstatic"] = semantics.get_linkstatic_default_for_test()

    _cc_test(**kwargs)
