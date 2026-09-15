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

"""Starlark implementation of cc_import.

We may change the implementation at any moment or even delete this file. Do not
rely on this. Pass the flag --experimental_starlark_cc_import
"""

load("@rules_cc//cc:cc_import.bzl", "cc_import")

def cc_import(**attrs):
    cc_import(**attrs)
