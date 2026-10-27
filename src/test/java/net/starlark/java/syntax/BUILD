load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/net/starlark/java:__pkg__"],
)

java_library(
    name = "TestUtils",
    srcs = ["TestUtils.java"],
    visibility = ["//src/test/java/net/starlark/java:__subpackages__"],
    deps = [
        "//src/main/java/net/starlark/java/syntax",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "SyntaxTests_lib",
    srcs = [
        "FileLocationsTest.java",
        "LValueBoundNamesTest.java",
        "LexerTest.java",
        "LocationTest.java",
        "NodePrinterTest.java",
        "NodeVisitorTest.java",
        "ParserInputTest.java",
        "ParserTest.java",
        "ProgramTest.java",
        "ResolverTest.java",
        "StarlarkFileTest.java",
        "TypeCheckerTest.java",
        "TypeTaggerTest.java",
        "TypesTest.java",
    ],
    deps = [
        ":TestUtils",
        "//src/main/java/net/starlark/java/syntax",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/junit",
        "//third_party/java/testparameterinjector",
        "//third_party/java/truth",
    ],
)

# SyntaxTests tests the Starlark parser/resolver.
java_test(
    name = "SyntaxTests",
    srcs = ["SyntaxTests.java"],  # suite
    deps = [
        ":SyntaxTests_lib",
        "//third_party/java/junit",
    ],
)
