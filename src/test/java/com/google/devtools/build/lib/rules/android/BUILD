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

java_test(
    name = "AndroidStarlarkCommonTest",
    srcs = ["AndroidStarlarkCommonTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_test(
    name = "AndroidConfigurationTest",
    srcs = ["AndroidConfigurationTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/rules/android",
        "//src/test/java/com/google/devtools/build/lib/analysis/util",
        "//third_party:junit4",
    ],
)
