licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

java_binary(
    name = "BazelPackagePrinter",
    srcs = ["BazelPackagePrinter.java"],
    deps = [
        ":Lib",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)

java_library(
    name = "Lib",
    srcs = ["Lib.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages",
    ],
)

sh_test(
    name = "bazel_package_printer_test",
    size = "medium",
    srcs = ["bazel_package_printer_test.sh"],
    data = [
        ":BazelPackagePrinter",
        "//src/test/shell/bazel:test-deps",
    ],
)
