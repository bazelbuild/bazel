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

"""C / C++"""
# Build Encyclopedia entry point for C / C++ rules implemented in Starlark in Blaze's @_builtins

binary_rules = struct(
    cc_binary = native.cc_binary,
)

library_rules = struct(
    cc_library = native.cc_library,
    cc_import = native.cc_import,
    cc_proto_library = native.cc_proto_library,
    cc_shared_library = native.cc_shared_library,
    # TODO: Replace with native.cc_static_library after bumping .bazelversion.
    **({"cc_static_library": native.cc_static_library} if hasattr(native, "cc_static_library") else {})
)

test_rules = struct(
    cc_test = native.cc_test,
)

other_rules = struct(
    fdo_prefetch_hints = native.fdo_prefetch_hints,
    fdo_profile = native.fdo_profile,
    memprof_profile = native.memprof_profile,
    propeller_optimize = native.propeller_optimize,
    cc_toolchain = native.cc_toolchain,
)
