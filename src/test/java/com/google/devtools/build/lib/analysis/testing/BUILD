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

java_library(
    name = "testing",
    testonly = 1,
    srcs = [
        "ExecGroupCollectionSubject.java",
        "ExecGroupSubject.java",
        "ResolvedToolchainContextSubject.java",
        "ToolchainCollectionSubject.java",
        "ToolchainContextSubject.java",
        "ToolchainInfoSubject.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:exec_group_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:resolved_toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/analysis:toolchain_collection",
        "//src/main/java/com/google/devtools/build/lib/analysis:toolchain_context",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:cmdline-primitives",
        "//src/main/java/com/google/devtools/build/lib/packages:exec_group",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:truth",
    ],
)
