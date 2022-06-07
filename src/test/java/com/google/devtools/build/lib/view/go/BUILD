# TODO(steinman): Go view test cases.
load("@rules_java//java:defs.bzl", "java_test")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "GoStarlarkApiTest",
    srcs = ["GoStarlarkApiTest.java"],
    tags = ["manual"],
    runtime_deps = ["//src/test/java/com/google/devtools/build/lib/analysis/util"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//src/test/java/com/google/devtools/build/lib/packages:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "GoCompileOnlyTest",
    srcs = ["GoCompileOnlyTest.java"],
    tags = ["manual"],
    runtime_deps = ["//src/test/java/com/google/devtools/build/lib/analysis/util"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
