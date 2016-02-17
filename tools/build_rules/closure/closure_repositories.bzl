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

CLOSURE_COMPILER_BUILD_FILE = """
java_import(
    name = "closure_compiler_jar",
    jars = ["compiler.jar"],
)

java_binary(
    name = "closure_compiler",
    main_class = "com.google.javascript.jscomp.CommandLineRunner",
    visibility = ["//visibility:public"],
    runtime_deps = [":closure_compiler_jar"],
)
"""

CLOSURE_LIBRARY_BUILD_FILE = """
load("@bazel_tools//tools/build_rules/closure:closure_js_library.bzl", "closure_js_library")
load("@bazel_tools//tools/build_rules/closure:closure_stylesheet_library.bzl", "closure_stylesheet_library")

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
"""

CLOSURE_TEMPLATES_BUILD_FILE = """
load("@bazel_tools//tools/build_rules/closure:closure_js_library.bzl", "closure_js_library")

java_import(
    name = "closure_templates_jar",
    jars = ["SoyToJsSrcCompiler.jar"],
)

java_binary(
    name = "closure_templates",
    main_class = "com.google.template.soy.SoyToJsSrcCompiler",
    visibility = ["//visibility:public"],
    runtime_deps = [":closure_templates_jar"],
)

closure_js_library(
    name = "closure_templates_js",
    srcs = ["soyutils_usegoog.js"],
    visibility = ["//visibility:public"],
)
"""

def closure_repositories():
  native.new_http_archive(
      name = "closure_compiler",
      build_file_content = CLOSURE_COMPILER_BUILD_FILE,
      sha256 = "e4e0cb49ad21ec26dd47693bdbd48f67aefe2d94fe8d9239312d2bcc74986538",
      url = "http://dl.google.com/closure-compiler/compiler-20150729.zip",
  )

  native.new_git_repository(
      name = "closure_library",
      build_file_content = CLOSURE_LIBRARY_BUILD_FILE,
      commit = "748b32441093c1474db2e0b3d074250e0bc47778",
      remote = "https://github.com/google/closure-library.git",
  )

  native.http_jar(
      name = "closure_stylesheets",
      sha256 = "8b2ae8ec3733171ec0c2e6536566df0b3c6da3e59b4784993bc9e73125d29c82",
      url = "https://closure-stylesheets.googlecode.com/files/closure-stylesheets-20111230.jar",
  )

  native.new_http_archive(
      name = "closure_templates",
      build_file_content = CLOSURE_TEMPLATES_BUILD_FILE,
      sha256 = "cdd94123cd0d1c3a183c15e855739c0aa5390297c22dddc731b8d7b23815e8a2",
      url = "http://dl.google.com/closure-templates/closure-templates-for-javascript-latest.zip",
  )

  native.bind(
      name = "closure_compiler_",
      actual = "@closure_compiler//:closure_compiler",
  )

  native.bind(
      name = "closure_templates_",
      actual = "@closure_templates//:closure_templates",
  )
