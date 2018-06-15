# Description:
#   Tools for android resource processing

package(default_visibility = [
    "//src/test/java/com/google/devtools/build/android:__subpackages__",
    "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
])

java_library(
    name = "resources",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:android_common_25_0_0",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/android/java/com/google/devtools/build/android:__pkg__"],
)
