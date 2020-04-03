load("@rules_java//java:defs.bzl", "java_library")

package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

# Redirection for util lib to allow exporting both ":util_internal" and the
# implementation of FileSystem as named in
# src/test/java/com/google/devtools/build/lib/testutil/TestConstants.java
java_library(
    name = "util",
    exports = [
        ":util_internal",
    ],
    runtime_deps = ["//src/main/java/com/google/devtools/build/lib/unix"],
)

java_library(
    name = "util_internal",
    testonly = 1,
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestConstants",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
    ],
)
