# -*- sh -*- (Bash only)
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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

# Bash completion of Bazel commands.
#
# Provides command-completion for:
# - bazel prefix options (e.g. --host_jvm_args)
# - blaze command-set (e.g. build, test)
# - blaze command-specific options (e.g. --copts)
# - values for enum-valued options
# - package-names, exploring all package-path roots.
# - targets within packages.

# Environment variables for customizing behavior:
#
# BAZEL_COMPLETION_USE_QUERY - if "true", `bazel query` will be used for
# autocompletion; otherwise, a heuristic grep is used. This is more accurate
# than the heuristic grep, especially for strangely-formatted BUILD files. But
# it can be slower, especially if the Bazel server is busy, and more brittle, if
# the BUILD file contains serious errors. This is an experimental feature.
#
# BAZEL_COMPLETION_ALLOW_TESTS_FOR_RUN - if "true", autocompletion results for
# `bazel run` includes test targets. This is convenient for users who run a lot
# of tests/benchmarks locally with 'bazel run'.

_bazel_completion_use_query() {
  _bazel__is_true "${BAZEL_COMPLETION_USE_QUERY}"
}

_bazel_completion_allow_tests_for_run() {
  _bazel__is_true "${BAZEL_COMPLETION_ALLOW_TESTS_FOR_RUN}"
}
