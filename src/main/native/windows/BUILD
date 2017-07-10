package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/native:__pkg__"],
)

# TODO(xingao): verify that this filegroup contains exactly what it has to, with
# regard to where it is used (//src/main/native:embedded_tools).
# Context: https://github.com/bazelbuild/bazel/commit/33d05f6b551cf2fdb257cb536a5e864d095144a1
filegroup(
    name = "embedded_tools",
    srcs = [":srcs"],
    visibility = ["//src/main/native:__pkg__"],
)

cc_library(
    name = "lib-file",
    srcs = ["file.cc"],
    hdrs = ["file.h"],
    visibility = [
        "//src/main/cpp:__subpackages__",
        "//src/test/cpp:__subpackages__",
        "//src/test/native:__subpackages__",
    ],
    deps = [":lib-util"],
)

cc_library(
    name = "lib-util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
)

genrule(
    name = "windows_jni",
    srcs = glob([
        "*.cc",
        "*.h",
    ]),
    outs = ["windows_jni.dll"],
    cmd = "$(location build_windows_jni.sh) $@ $(SRCS)",
    tools = ["build_windows_jni.sh"],
    visibility = [
        "//src:__pkg__",
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android:__subpackages__",
    ],
)
