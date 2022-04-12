load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "action_graph_query_helper",
    testonly = 1,
    srcs = ["ActionGraphQueryHelper.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target_value",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/test/java/com/google/devtools/build/lib/query2/testutil",
        "//third_party:guava",
    ],
)

java_library(
    name = "action_graph_query_test",
    testonly = 1,
    srcs = ["ActionGraphQueryTest.java"],
    deps = [
        ":action_graph_query_helper",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/build_configuration",
        "//src/main/java/com/google/devtools/build/lib/analysis:configured_target_value",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/test/java/com/google/devtools/build/lib/query2/testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
