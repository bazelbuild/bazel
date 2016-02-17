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
      sha256 = "215ba5df026e5d92bda6634463a9c634d38a1aa4b6dab336da5c52e884cbde95",
      url = "https://dl.google.com/closure-compiler/compiler-latest.zip",
  )

  native.new_http_archive(
      name = "closure_library",
      build_file_content = CLOSURE_LIBRARY_BUILD_FILE,
      sha256 = "8f610300e4930190137505a574a54d12346426f2a7b4f179026e41674e452a86",
      url = "https://github.com/google/closure-library/archive/20160208.zip",
  )

  native.http_jar(
      name = "closure_stylesheets",
      sha256 = "5308cb46f7677b9995237ade57770d27592aff69359d29be571220a2bf10e724",
      url = "https://github.com/google/closure-stylesheets/releases/download/v1.1.0/closure-stylesheets.jar",
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
