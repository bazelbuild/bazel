# Copyright 2015 Google Inc. All rights reserved.
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

objc_binary(
    name = "PrenotCalculator",
    srcs = [
        "PrenotCalculator/main.m",
    ],
    infoplist = "PrenotCalculator/PrenotCalculator-Info.plist",
    visibility = ["//visibility:public"],
    deps = [
        ":PrenotCalculatorClasses",
    ],
)

objc_library(
    name = "PrenotCalculatorClasses",
    srcs = [
        "PrenotCalculator/AppDelegate.m",
        "PrenotCalculator/CalculatedValues.m",
        "PrenotCalculator/CalculatorViewController.m",
        "PrenotCalculator/CoreData.m",
        "PrenotCalculator/Equation.m",
        "PrenotCalculator/Literal.m",
        "PrenotCalculator/ValuesViewController.m",
    ],
    hdrs = glob(
        ["**/*.h"],
        exclude = ["PrenotCalculator/Expression.h"],
    ),
    bundles = [":PrenotCalculatorResources"],
    sdk_frameworks = ["CoreData"],
    xibs = ["PrenotCalculator/CalculatorViewController.xib"],
    deps = [
        ":CoreDataResources",
        ":ExpressionPrebuilt",
    ],
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
# depend on that instead. Instructions:
# Build using the following command repeatedly with the desired CPU values:
#   bazel build --ios_cpu=<cpu> examples/objc/PrenotCalculator:ExpressionClasses
# and combining the resulting .a's (copied after each bazel build) using lipo:
#   /usr/bin/lipo -create expression_i386.a expression_armv7.a -output expression_prebuilt.a
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

objc_library(
    name = "CoreDataResources",
    datamodels = glob(["PrenotCalculator/DataModel.xcdatamodeld/**"]),
)

# If instruments is supported this genrule should *not timeout* and it should
# produce a screenshot of the PrenotCalculator.ipa in ~ few seconds.
genrule(
    name = "PrenotCalculatorInstruments",
    srcs = [":PrenotCalculator.ipa"],
    outs = ["hello_instruments.png"],
    cmd =
        # This command expects the PrenotCalculator.ipa as input then unzips
        # it into a temporary directory, creates a .js script on the fly
        # that captures a screenshot, and finally invokes /usr/bin/instruments
        # with the former as arguments.
        "unzip -q $(location :PrenotCalculator.ipa) -d $${TMPDIR} ; " +
        "echo 'UIATarget.localTarget().delay(5); UIATarget.localTarget().captureScreenWithName(\"hello_instruments\");' > $${TMPDIR}/hello_instruments.js && " +
        "/usr/bin/instruments " +
        "-t /Applications/Xcode.app/Contents/Applications/Instruments.app/Contents/PlugIns/AutomationInstrument.xrplugin/Contents/Resources/Automation.tracetemplate " +
        "-w 'iPhone 5 (8.1 Simulator)' " +
        "$${TMPDIR}/Payload/PrenotCalculator.app " +
        "-e UIASCRIPT $${TMPDIR}/hello_instruments.js " +
        "-e UIARESULTSPATH $${TMPDIR} -v && " +
        "cp $${TMPDIR}'/Run 1/hello_instruments.png' $(@)",
    tags = ["requires-darwin"],
)

filegroup(
    name = "srcs",
    srcs = glob([
        "**",
        "BUILD",
    ]),
    visibility = ["//examples:__pkg__"],
)
