# Copyright 2015 The Bazel Authors. All rights reserved.
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

# BUILD file to use the Java AppEngine SDK with a remote repository.
java_import(
    name = "jars",
    jars = glob(["**/*.jar"]),
    visibility = ["//visibility:public"],
)

java_import(
    name = "api",
    jars = ["appengine-java-sdk-1.9.23/lib/impl/appengine-api.jar"],
    visibility = ["//visibility:public"],
    neverlink = 1,
)

filegroup(
    name = "sdk",
    srcs = glob(["appengine-java-sdk-1.9.23/**"]),
    visibility = ["//visibility:public"],
    path = "appengine-java-sdk-1.9.23",
)