load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

WINDOWS_SRCS = [
    "WindowsSubprocess.java",
    "WindowsSubprocessFactory.java",
]

# Library for dealing with executable commands, including their arguments and runtime environment
# (environment variables, working directory). It lets a caller execute a command, get its results,
# and optionally forward interrupts to the subprocess. The library also handles creating threads to
# ensure timely reading of subprocess outputs.
java_library(
    name = "shell",
    srcs = glob(
        ["*.java"],
        exclude = WINDOWS_SRCS,
    ),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker:__pkg__",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:TestType",
        "//src/main/java/com/google/devtools/build/lib/util:describable_execution_unit",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:execution_statistics_java_proto",
        "//third_party/java/auto:auto_value",
        "//third_party/java/flogger",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/guava:io",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "windows_subprocess_factory",
    srcs = WINDOWS_SRCS,
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker:__pkg__",
    ],
    deps = [
        ":shell",
        "//src/main/java/com/google/devtools/build/lib/util:cleaner",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/windows:processes",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
    ],
)
