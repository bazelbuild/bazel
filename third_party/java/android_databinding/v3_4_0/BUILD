# Description:
#   Android data binding buildtime libs and common lib shared with Android
#   clients: implementation.

load("@rules_java//java:defs.bzl", "java_import")

package(default_visibility = ["//third_party/java/android_databinding:__pkg__"])

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

# Resource processor implementation.
# Built from https://cs.android.com/androidx/platform/frameworks/data-binding/+/mirror-goog-studio-master-dev:exec/src
java_import(
    name = "exec",
    jars = ["exec.jar"],
    deps = ["@maven_android//:androidx_databinding_databinding_compiler"],
)
