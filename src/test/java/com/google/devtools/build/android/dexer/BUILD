load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/android:__pkg__"],
)

# Description:
#   Tests for the DexFileSplitter.
java_library(
    name = "tests",
    srcs = [
        "AllTests.java",
    ] + select({
        "//external:has_androidsdk": ["DexFileSplitterTest.java"],
        "//conditions:default": ["NoAndroidSdkStubTest.java"],
    }),
    javacopts = ["-source 8 -target 8"],
    resources = ["testresource.txt"],
    deps = [
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestSuite",
        "//src/tools/android/java/com/google/devtools/build/android/dexer:DexFileSplitter_lib",
        "//src/tools/android/java/com/google/devtools/build/android/r8:dexbuilder_deploy.jar",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
        "@bazel_tools//tools/java/runfiles",
    ] + select({
        "//external:has_androidsdk": ["//external:android/dx_jar_import"],
        "//conditions:default": [],
    }),
)

java_library(
    name = "testdata",
    srcs = glob(["testdata/**/*.java"]),
)

java_test(
    name = "AllTests",
    size = "small",
    data = [
        "test_main_dex_list.txt",
        ":mixed_testinput",
        ":testdata",
        ":tests",
    ],
    jvm_flags = [
        "-Dtestmaindexlist=io_bazel/$(location :test_main_dex_list.txt)",
        "-Dtestinputjar=io_bazel/$(location :tests)",
        "-Dtestinputjar2=io_bazel/$(location :testdata)",
        "-Dmixedinputjar=io_bazel/$(location :mixed_testinput)",
    ],
    runtime_deps = [
        ":tests",
    ],
)

genrule(
    name = "mixed_srcjar_gen",
    outs = ["mixed.srcjar"],
    cmd = """set -eu
        readonly tmpdir="$$(mktemp -d)"
        echo "package j$$.test; public class Foo {}" > "$${tmpdir}/Foo.java"
        echo "package zzz; public class Bar {}" > "$${tmpdir}/Bar.java"
        echo "package aaa; public class Baz {}" > "$${tmpdir}/Baz.java"
        $(location //tools/zip:zipper) c $@ \\
            "j$$/test/Foo.java=$${tmpdir}/Foo.java" \\
            "zzz/Bar.java=$${tmpdir}/Bar.java" \\
            "aaa/Baz.java=$${tmpdir}/Baz.java"
        rm -rf "$${tmpdir}" """,
    tools = ["//tools/zip:zipper"],
)

java_library(
    name = "mixed_testinput",
    srcs = [":mixed_srcjar_gen"],
)
