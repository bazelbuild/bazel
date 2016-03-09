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

ios_application(
    name = "PrenotCalculator",
    binary = ":PrenotCalculator_binary",
    bundles = [":PrenotCalculatorResources"],
    datamodels = glob(["PrenotCalculator/DataModel.xcdatamodeld/**"]),
    infoplist = "PrenotCalculator/PrenotCalculator-Info.plist",
    visibility = ["//visibility:public"],
    xibs = ["PrenotCalculator/CalculatorViewController.xib"],
)

objc_binary(
    name = "PrenotCalculator_binary",
    srcs = [
        "PrenotCalculator/AppDelegate.m",
        "PrenotCalculator/CalculatedValues.m",
        "PrenotCalculator/CalculatorViewController.m",
        "PrenotCalculator/CoreData.m",
        "PrenotCalculator/Equation.m",
        "PrenotCalculator/Literal.m",
        "PrenotCalculator/ValuesViewController.m",
        "PrenotCalculator/main.m",
    ],
    hdrs = glob(
        ["**/*.h"],
        exclude = ["PrenotCalculator/Expression.h"],
    ),
    sdk_frameworks = ["CoreData"],
    deps = [":ExpressionPrebuilt"],
)

# A prebuilt library that contains multiple architectures.
# Currently compiled for: i386, x86_64, armv7.
objc_import(
    name = "ExpressionPrebuilt",
    hdrs = ["PrenotCalculator/Expression.h"],
    archives = [
        "expression_prebuilt.a",
    ],
)

# Don't use this directly, instead use it to construct ExpressionPrebuilt and
# depend on that instead.
# You can build using the following command:
#    for i in armv7 arm64 i386 x86_64; \
#    do output/bazel build --xcode_version=7.2 --ios_sdk_version=9.2 --ios_cpu=$i examples/objc:ExpressionClasses; \
#    mv bazel-bin/examples/objc/libExpressionClasses.a bazel-bin/examples/objc/libExpressionClasses_$i.a; \
#    done; \
#    xcrun lipo -create bazel-bin/examples/objc/libExpressionClasses_{armv7,arm64,i386,x86_64}.a -output bazel-bin/examples/objc/expression_prebuilt.a \
#    cp bazel-bin/examples/objc/expression_prebuilt.a examples/objc/expression_prebuilt.a
# You may need to adjust the xcode_version and ios_sdk_versions as appropriate.
objc_library(
    name = "ExpressionClasses",
    srcs = [
        "PrenotCalculator/Expression.m",
    ],
    hdrs = ["PrenotCalculator/Expression.h"],
)

objc_bundle_library(
    name = "PrenotCalculatorResources",
    resources = glob(["PrenotCalculator/Resources/**"]),
)

filegroup(
    name = "srcs",
    srcs = glob([
        "**",
        "BUILD",
    ]),
    visibility = ["//examples:__pkg__"],
)
