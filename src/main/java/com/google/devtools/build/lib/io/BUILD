load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# TODO(janakr): move other general-purpose IO classes here (FileSymlinkException
#  and friends, for instance).
java_library(
    name = "inconsistent_filesystem_exception",
    srcs = ["InconsistentFilesystemException.java"],
)
