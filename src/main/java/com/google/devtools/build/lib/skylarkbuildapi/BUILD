# Description:
#   This package contains interfaces representing the skylark "build API"
#   (but not the implementation of that API). Ultimately, this package
#   may be broken out of the Bazel package hierarchy to be standalone.
#   Thus, this package should not depend on Bazel-specific packages (only
#   those which contain pure-Skylark concepts, such as the interpreter or
#   annotation interfaces).

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/android:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/python:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/repository:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/proto:srcs",
    ],
)

java_library(
    name = "skylarkbuildapi",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:skylark_semantics",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
