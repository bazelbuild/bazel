# Copyright 2015 Google Inc. All Rights Reserved.
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
