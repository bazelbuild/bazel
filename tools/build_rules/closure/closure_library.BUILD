# Copyright 2015 The Bazel Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("/tools/build_rules/closure/closure_js_library", "closure_js_library")
load("/tools/build_rules/closure/closure_stylesheet_library", "closure_stylesheet_library")

closure_js_library(
    name = "closure_library",
    srcs = glob(
        [
            "closure/goog/**/*.js",
            "third_party/closure/goog/**/*.js",
        ],
        exclude = [
            "closure/goog/**/*_test.js",
            "closure/goog/demos/**/*.js",
            "third_party/closure/goog/**/*_test.js",
        ],
    ),
    visibility = ["//visibility:public"],
)

closure_stylesheet_library(
    name = "closure_library_css",
    srcs = glob(["closure/goog/css/**/*.css"]),
    visibility = ["//visibility:public"],
)
