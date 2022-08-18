# Description:
#   Tool for desugaring Java constructs not supported by Android tools or devices.

filegroup(
    name = "embedded_tools",
    srcs = ["BUILD.tools"],
    visibility = [
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
    ],
)

java_library(
    name = "desugar",
    visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android:__pkg__",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:__pkg__",
    ],
    exports = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/dependencies:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/nest:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/covariantreturn:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/retarget:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/typeannotation:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/typehierarchy:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/config:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/corelibadapter:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/runtime:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/strconcat:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/preanalysis:srcs",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/scan:srcs",
    ],
    visibility = ["//src/tools/android/java/com/google/devtools/build/android:__pkg__"],
)
