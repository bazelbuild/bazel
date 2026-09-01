load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "noop_connectivity",
    srcs = ["NoOpConnectivityModule.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib:__subpackages__",
        "//src/test/java/com/google/devtools/build/lib/buildtool:__subpackages__",
    ],
    deps = [
        ":connectivity_status",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
    ],
)

java_library(
    name = "connectivity_status",
    srcs = [
        "ConnectivityStatus.java",
        "ConnectivityStatusProvider.java",
    ],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/buildeventservice:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/buildeventservice:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/buildtool/util:__subpackages__",
    ],
)
