# Copyright 2017 The Bazel Authors. All rights reserved.
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

"""Bazel-specific intellij aspect."""

load("@bazel_tools//tools/ide:intellij_info_impl.bzl",
     "make_intellij_info_aspect",
     "intellij_info_aspect_impl")

def tool_label(label_str):
  """Returns a label that points to a bazel tool."""
  return Label("@bazel_tools" + label_str)

semantics = struct(
    tool_label = tool_label,
)

def _aspect_impl(target, ctx):
  return intellij_info_aspect_impl(target, ctx, semantics)

intellij_info_aspect = make_intellij_info_aspect(_aspect_impl, semantics)
