# Copyright 2019 The Bazel Authors. All rights reserved.
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
"""Temporary rule that can only fail with a nice error message."""

def _impl(ctx):
    fail("Constraints from @bazel_tools//platforms have been removed. " +
         "Please use constraints from @platforms repository embedded in " +
         "Bazel, or preferably declare dependency on " +
         "https://github.com/bazelbuild/platforms. See " +
         "https://github.com/bazelbuild/bazel/issues/8622 for details.")

fail_with_incompatible_use_platforms_repo_for_constraints = rule(
    implementation = _impl,
)
