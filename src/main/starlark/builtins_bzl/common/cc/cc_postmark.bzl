# Copyright 2025 The Bazel Authors. All rights reserved.
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

"""PostMark is not implemented in Bazel."""

def _get_postmark_attrs():
    return {}

def _postmark_initializer(**kwargs):
    return kwargs

def _add_postmark_action(
        ctx,  # @unused
        cc_toolchain,  # @unused
        binary,  # @unused
        output_binary_for_linking):  # @unused
    pass

def _get_use_postmark(ctx):  # @unused
    return False

postmark = struct(
    get_attrs = _get_postmark_attrs,
    add_action = _add_postmark_action,
    get_use_postmark = _get_use_postmark,
    initializer = _postmark_initializer,
)
