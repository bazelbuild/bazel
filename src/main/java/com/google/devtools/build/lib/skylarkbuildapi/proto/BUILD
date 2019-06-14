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
    srcs = glob(["**"]),
)

java_library(
    name = "proto",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
