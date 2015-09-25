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

java_library(
    name = "J2ObjcExample-Java",
    srcs = glob(["src/main/java/**/*.java"]),
    deps = [
        "//third_party/java/j2objc:annotations",
    ],
)

j2objc_library(
    name = "J2ObjcExample-J2Objc",
    entry_classes = ["com.example.myproject.SimpleGreeter"],
    deps = [":J2ObjcExample-Java"],
)

objc_binary(
    name = "J2ObjcExample",
    srcs = glob([
        "J2ObjcExample/*.m",
    ]),
    hdrs = glob([
        "J2ObjcExample/*.h",
    ]),
    infoplist = "J2ObjcExample/J2ObjcExample-Info.plist",
    deps = [
        ":J2ObjcExample-J2Objc",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "J2ObjcExample/*",
        "src/**",
        "BUILD",
    ]),
    visibility = ["//examples:__pkg__"],
)
