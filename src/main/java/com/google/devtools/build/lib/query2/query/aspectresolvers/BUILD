load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "aspectresolvers",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib:build_lib",
        "//src/main/java/com/google/devtools/build/lib:full_access",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//third_party/java/guava:collect",
    ],
)
