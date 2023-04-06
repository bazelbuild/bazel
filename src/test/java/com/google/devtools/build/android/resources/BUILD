load("@rules_java//java:defs.bzl", "java_test")

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

java_test(
    name = "RClassGeneratorTest",
    size = "small",
    srcs = glob([
        "**/*.java",
    ]),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
        "//src/tools/android/java/com/google/devtools/build/android/resources",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
