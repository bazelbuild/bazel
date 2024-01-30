load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "cquery-node",
    srcs = [
        "CqueryNode.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:action_lookup_key",
        "//src/main/java/com/google/devtools/build/lib/analysis:config/config_matching_provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages:label_printer",
        "//src/main/java/com/google/devtools/build/lib/skyframe/config",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "abstract-blaze-query-env",
    srcs = [
        "AbstractBlazeQueryEnvironment.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:label_printer",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "options",
    srcs = [
        "CommonQueryOptions.java",
        "FlagConstants.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/packages:label_printer",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/query2/query/aspectresolvers",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/net/starlark/java/eval",
        "//third_party:guava",
    ],
)

java_library(
    name = "QueryTransitivePackagePreloader",
    srcs = [
        "QueryTransitivePackagePreloader.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/skyframe:detailed_exceptions",
        "//src/main/java/com/google/devtools/build/lib/skyframe:transitive_target_key",
        "//src/main/java/com/google/devtools/build/lib/skyframe:transitive_target_value",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "universe-scope",
    srcs = [
        "UniverseScope.java",
    ],
    deps = [
        ":UniverseSkyKey",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/skyframe:prepare_deps_of_patterns_value",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "UniverseSkyKey",
    srcs = ["UniverseSkyKey.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//third_party:guava",
    ],
)
