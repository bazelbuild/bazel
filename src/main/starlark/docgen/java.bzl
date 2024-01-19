# Copyright 2024 The Bazel Authors. All rights reserved.
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

"""Java"""
# Build Encyclopedia entry point for Java rules implemented in Starlark in Bazel's @_builtins

binary_rules = struct(
)

library_rules = struct(
    java_import = native.java_import,
    java_library = native.java_library,
    java_lite_proto_library = native.java_lite_proto_library,
    java_proto_library = native.java_proto_library,
)

test_rules = struct(
    java_test = native.java_test,
)

other_rules = struct(
    java_package_configuration = native.java_package_configuration,
    java_plugin = native.java_plugin,
    java_runtime = native.java_runtime,
    java_toolchain = native.java_toolchain,
)
