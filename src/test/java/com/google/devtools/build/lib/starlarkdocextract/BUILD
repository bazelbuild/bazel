load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/docgen:__pkg__",
        "//src/test/java/com/google/devtools/build/lib:__pkg__",
    ],
)

java_test(
    name = "LabelRendererTest",
    size = "small",
    srcs = ["LabelRendererTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:labelrenderer",
        "//src/main/java/net/starlark/java/eval",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "StarlarkFunctionInfoExtractorTest",
    size = "small",
    srcs = ["StarlarkFunctionInfoExtractorTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:extractionexception",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:labelrenderer",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:starlarkfunctioninfoextractor",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//src/test/java/com/google/devtools/build/lib/starlark/util",
        "//third_party/java/junit",
        "//third_party/java/truth",
        "//third_party/java/truth/extensions:proto",
    ],
)

java_test(
    name = "RuleInfoExtractorTest",
    srcs = ["RuleInfoExtractorTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:labelrenderer",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:ruleinfoextractor",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//src/test/java/com/google/devtools/build/lib/packages:PackageTestsUtil",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

java_test(
    name = "ModuleInfoExtractorTest",
    srcs = ["ModuleInfoExtractorTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe:skyframe_cluster",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:extractionexception",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:labelrenderer",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:moduleinfoextractor",
        "//src/main/java/com/google/devtools/build/lib/starlarkdocextract:ruleinfoextractor",
        "//src/main/java/net/starlark/java/eval",
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/protobuf:stardoc_output_java_proto",
        "//src/test/java/com/google/devtools/build/lib/starlark/util",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
        "//third_party/java/truth/extensions:proto",
    ],
)
