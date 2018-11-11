# Description:
#   Skydoc is a documentation generator for skylark files.
#
#   It is currently experimental and in development.
#
# Usage:
# skydoc <target_file> <output_file>

package(
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/android:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/apple:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/config:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/cpp:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/java:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/platform:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/repository:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/test:srcs",
        "//src/main/java/com/google/devtools/build/skydoc/rendering:srcs",
    ],
)

java_binary(
    name = "skydoc",
    jvm_flags = [
        # quiet warnings from com.google.protobuf.UnsafeUtil,
        # see: https://github.com/google/protobuf/issues/3781
        # and: https://github.com/bazelbuild/bazel/issues/5599
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        # ... but only on JDK >= 9
        "-XX:+IgnoreUnrecognizedVMOptions",
    ],
    main_class = "com.google.devtools.build.skydoc.SkydocMain",
    visibility = ["//visibility:public"],
    runtime_deps = [
        ":skydoc_lib",
    ],
)

java_library(
    name = "skydoc_lib",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/android",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/apple",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/config",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/cpp",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/java",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/repository",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/android",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/apple",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/config",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/cpp",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/java",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/platform",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/repository",
        "//src/main/java/com/google/devtools/build/skydoc/fakebuildapi/test",
        "//src/main/java/com/google/devtools/build/skydoc/rendering",
        "//third_party:guava",
    ],
)
