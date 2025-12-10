load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# SyntaxTests tests the Starlark parser/resolver.
java_test(
    name = "SyntaxTests",
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
        "SyntaxTests.java",  # (suite)
        "TypeResolverTest.java",
        "TypesTest.java",
    ],
    deps = [
        "//src/main/java/net/starlark/java/syntax",
        "//src/main/java/net/starlark/java/types",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:junit4",
        "//third_party:truth",
        "@maven//:com_google_testparameterinjector_test_parameter_injector",
    ],
)
