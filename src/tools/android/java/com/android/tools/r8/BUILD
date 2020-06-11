# Description:
#   Gateway classes to access package private parts of classes in com.android.tools.r8.

load("@rules_java//java:defs.bzl", "java_library")

licenses(["notice"])

package(default_visibility = ["//javatests/com/google/devtools/build/android:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src/java/com/google/devtools/build/android/r8:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
)

# Expose this as a separate target because Bazel requires all java_libraries that aren't pure exports to have sources.
java_library(
    name = "compat_dx_support",
    srcs = ["CompatDxSupport.java"],
    deps = ["//external:android/d8_jar_import"],
)

java_library(
    name = "r8_support",
    visibility = [
        "//src/test/java/com/google/devtools/build/android/r8:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android/r8:__pkg__",
    ],
    exports = select({
        "//external:has_androidsdk": [
            ":compat_dx_support",
        ],
        "//conditions:default": [],
    }),
)
