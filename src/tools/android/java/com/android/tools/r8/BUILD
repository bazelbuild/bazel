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

java_library(
    name = "r8_support",
    srcs = select({
        "//external:has_androidsdk": glob(
            ["*.java"],
        ),
        "//conditions:default": [],
    }),
    visibility = [
        "//src/test/java/com/google/devtools/build/android/r8:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android/r8:__pkg__",
    ],
    deps = select({
        "//external:has_androidsdk": ["//external:android/d8_jar_import"],
        "//conditions:default": [],
    }),
)
