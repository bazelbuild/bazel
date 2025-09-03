load("@rules_java//java:defs.bzl", "java_library")

# Description:
#   SingleJar combines multiple zip files and additional files
#   into a single zip file.
package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/singlejar:singlejar_package_group"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

# This is used as a library outside of Bazel.
java_library(
    name = "libSingleJar",
    srcs = [
        "ConcatenateStrategy.java",
        "CopyEntryFilter.java",
        "DefaultJarEntryFilter.java",
        "JarUtils.java",
        "ZipCombiner.java",
        "ZipEntryFilter.java",
    ],
    visibility = [
        "//src/java_tools/singlejar:__pkg__",
        "//src/java_tools/singlejar:singlejar_package_group",
        "//src/java_tools/singlejar/javatests/com/google/devtools/build/singlejar:__pkg__",
        "//src/test/java/com/google/devtools/build/android:__pkg__",
        "//third_party/bazel_rules/rules_android/src/tools/java/com/google/devtools/build/android:__pkg__",
        "//third_party/bazel_rules/rules_android/src/tools/javatests/com/google/devtools/build/android:__pkg__",
        "//third_party/bazel_rules/rules_android/src/tools/javatests/com/google/devtools/build/android/r8:__pkg__",
    ],
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//third_party:jsr305",
    ],
)
