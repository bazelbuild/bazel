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
    cc_proto_library = native.cc_proto_library,
)

test_rules = struct(
    cc_test = native.cc_test,
)

other_rules = struct(
)
