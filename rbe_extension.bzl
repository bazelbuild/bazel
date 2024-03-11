# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Module extensions for loading RBE toolchains.

"""

load("@bazelci_rules//:rbe_repo.bzl", "rbe_preconfig")

def _bazel_rbe_deps(_ctx):
    rbe_preconfig(
        name = "rbe_ubuntu2004",
        toolchain = "ubuntu2004",
    )

bazel_rbe_deps = module_extension(implementation = _bazel_rbe_deps)
