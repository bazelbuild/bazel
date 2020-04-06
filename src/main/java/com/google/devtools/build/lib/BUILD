load("@rules_java//java:defs.bzl", "java_library")
load(":merge_licenses.bzl", "merge_licenses")

########################################################################
# Description:
#   Main Java code for Bazel
#
# IMPORTANT:  NOT A PUBLIC INTERFACE.
#
# DO NOT ADD A DEPENDENCY on any targets or sub-packages outside of
# __subpackages__ without first talking to Bazel maintainers.
#
# Here are two good reasons:
#
# 1.  FRAGILITY. This is not a stable API, it's an internal interface.
#     It changes from time to time, and your client *will* be broken.
#
# 2.  CORRECTNESS.  The BUILD file parser changes rapidly.  If
#     your application depends upon it, your application will start to
#     bit-rot almost immediately.
#
# There are better alternatives.  The 'bazel query' tool has
# stable command-line and programmatic interfaces that will work with any
# version of Bazel.  They are also debugged and documented, so you'll
# avoid many common traps and pitfalls when (e.g.)  computing
# dependencies.
package(
    default_visibility = ["//src:__subpackages__"],
)

# Generate list of all srcs via:
# bazel query 'filter("srcs", kind("filegroup rule", //src/main/java/com/google/devtools/build/lib/...))' | sort | sed -e "s/^/\"/" | sed -e "s/$/\",/" | fgrep -v "build/lib:srcs"
filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/docgen:srcs",
        "//src/main/java/com/google/devtools/build/lib/actions:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations:srcs",
        "//src/main/java/com/google/devtools/build/lib/authandtls:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream:srcs",
        "//src/main/java/com/google/devtools/build/lib/causes:srcs",
        "//src/main/java/com/google/devtools/build/lib/clock:srcs",
        "//src/main/java/com/google/devtools/build/lib/cmdline:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashmap:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect:srcs",
        "//src/main/java/com/google/devtools/build/lib/concurrent:srcs",
        "//src/main/java/com/google/devtools/build/lib/dynamic:srcs",
        "//src/main/java/com/google/devtools/build/lib/events:srcs",
        "//src/main/java/com/google/devtools/build/lib/exec/local:srcs",
        "//src/main/java/com/google/devtools/build/lib/graph:srcs",
        "//src/main/java/com/google/devtools/build/lib/metrics:srcs",
        "//src/main/java/com/google/devtools/build/lib/network:srcs",
        "//src/main/java/com/google/devtools/build/lib/packages:srcs",
        "//src/main/java/com/google/devtools/build/lib/pkgcache:srcs",
        "//src/main/java/com/google/devtools/build/lib/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler/callcounts:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/common:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/compat:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/engine:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/query/aspectresolvers:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2/query/output:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/android:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/swift:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genrule:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/objc:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/python:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/lib/shell:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/module:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/server:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface/processor:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface:srcs",
        "//src/main/java/com/google/devtools/build/lib/ssd:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/main/java/com/google/devtools/build/lib/supplier:srcs",
        "//src/main/java/com/google/devtools/build/lib/syntax:srcs",
        "//src/main/java/com/google/devtools/build/lib/unix:srcs",
        "//src/main/java/com/google/devtools/build/lib/unsafe:srcs",
        "//src/main/java/com/google/devtools/build/lib/util:srcs",
        "//src/main/java/com/google/devtools/build/lib/versioning:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/windows/jni:srcs",
        "//src/main/java/com/google/devtools/build/lib/windows:srcs",
        "//src/main/java/com/google/devtools/build/lib/worker:srcs",
        "//src/main/java/com/google/devtools/build/skyframe:srcs",
        "//src/main/java/com/google/devtools/common/options:srcs",
    ],
    visibility = [
        "//src:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

########################################################################
#
# The "foundation" library (concurrent, events, util, vfs, inmemoryfs, options)
# Filesystem abstraction, event reporting, options parsing, I/O routines.
#

# These packages apparently depend on :foundation. I have no idea why, but to
# avoid breaking them, I allow them to depend on it.
java_library(
    name = "foundation",
    exports = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

# Skylark language for configuration files.
#
# IMPORTANT:  NOT A PUBLIC INTERFACE. TARGETS SHOULDN'T DEPEND ON THIS.
#
java_library(
    name = "skylark-lang",
    visibility = ["//visibility:public"],
    exports = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
    ],
)

# Provides core configuration transition abstractions and Blaze-common (i.e not
# rule-specific) transitions.
java_library(
    name = "config-transitions",
    srcs = glob([
        "analysis/config/transitions/*.java",
    ]) + [
        "analysis/config/BuildOptions.java",
        "analysis/config/FragmentOptions.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "syntax",
    srcs = [],
    visibility = ["//visibility:public"],
    exports = [
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
    ],
)

java_library(
    name = "transitive-info-provider",
    srcs = ["analysis/TransitiveInfoProvider.java"],
)

java_library(
    name = "config-matching-provider",
    srcs = ["analysis/config/ConfigMatchingProvider.java"],
    deps = [
        ":transitive-info-provider",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
    ],
)

java_library(
    name = "provider-collection",
    srcs = [
        "analysis/ProviderCollection.java",
    ],
    deps = [
        ":transitive-info-provider",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "build-configuration-option-details",
    srcs = ["analysis/config/BuildConfigurationOptionDetails.java"],
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/rules/config:__pkg__",
    ],
    deps = [
        ":build-base",
    ],
)

java_library(
    name = "build-base",
    srcs = glob(
        [
            "analysis/*.java",
            "analysis/actions/*.java",
            "analysis/buildinfo/*.java",
            "analysis/config/*.java",
            "analysis/config/transitions/*.java",
            "analysis/configuredtargets/*.java",
            "analysis/constraints/*.java",
            "analysis/extra/*.java",
            "analysis/fileset/*.java",
            "analysis/skylark/*.java",
            "analysis/stringtemplate/*.java",
            "analysis/test/*.java",
            "exec/*.java",
            "repository/*.java",
            "rules/*.java",
            "rules/extra/*.java",
            "rules/filegroup/*.java",
            "rules/fileset/*.java",
            "rules/repository/*.java",
            "skyframe/*.java",
            "skyframe/actiongraph/*.java",
            "skyframe/actiongraph/v2/*.java",
        ],
        allow_empty = True,
        exclude = [
            "analysis/BuildInfo.java",
            "analysis/BuildVersionInfo.java",
            "analysis/ProviderCollection.java",
            "analysis/TransitiveInfoProvider.java",
            "analysis/config/BuildConfigurationOptionDetails.java",
            "rules/repository/CoreWorkspaceRules.java",
        ],
    ) + [
        "runtime/BlazeServerStartupOptions.java",
        "runtime/BuildEventStreamerUtils.java",
    ],
    exports = [
        ":rule-definition-context",
        ":transitive-info-provider",
    ],
    deps = [
        ":bug-report",
        ":build-request-options",
        ":keep-going-option",
        ":provider-collection",
        ":transitive-info-provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/actionsketch:action_sketch",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:platform_utils",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/cmdline:LabelValidator",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib/query2:aquery-utils",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:cpp_interface",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe/proto:action_rewind_event_java_proto",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/config",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/core",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/supplier",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/unix/jni",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:command",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:process",
        "//src/main/java/com/google/devtools/build/lib/util:resource_converter",
        "//src/main/java/com/google/devtools/build/lib/util:resource_usage",
        "//src/main/java/com/google/devtools/build/lib/util:shell_escaper",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:output_service",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:analysis_java_proto",
        "//src/main/protobuf:analysis_v2_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:spawn_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//src/main/protobuf:xcode_java_proto",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
        "//third_party/protobuf:protobuf_java_util",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

# Rules
java_library(
    name = "nativedeps-rules",
    srcs = glob(
        ["rules/nativedeps/*.java"],
    ),
    deps = [
        ":build-base",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "core-rules",
    srcs = glob(
        ["rules/core/*.java"],
    ),
    deps = [
        ":build-base",
        "//third_party:guava",
    ],
)

# TODO(bazel-team): Include in the srcs the implementations of the rules inside
# CoreWorkspaceRules.java once they can be extracted from the "build-base" target. They can't be
# extracted currently because of non-trivial inter-package dependencies.
java_library(
    name = "core-workspace-rules",
    srcs = [
        "rules/repository/CoreWorkspaceRules.java",
    ],
    deps = [
        ":build-base",
        ":core-rules",
        "//third_party:guava",
    ],
)

java_library(
    name = "testing-support-rules",
    srcs = glob(
        ["rules/test/*.java"],
    ),
    deps = [
        ":build-base",
        ":core-rules",
        ":transitive-info-provider",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_proto",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "shared-base-rules",
    exports = [
        ":nativedeps-rules",
        "//src/main/java/com/google/devtools/build/lib/rules/android",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/rules/proto",
        "//src/main/java/com/google/devtools/build/lib/rules/python",
    ],
)

java_library(
    name = "build-request-options",
    srcs = ["buildtool/BuildRequestOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:resource_converter",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "keep-going-option",
    srcs = ["runtime/KeepGoingOption.java"],
    deps = ["//src/main/java/com/google/devtools/common/options"],
)

java_library(
    name = "loading-phase-threads-option",
    srcs = ["runtime/LoadingPhaseThreadsOption.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/util:resource_converter",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "server",
    srcs = [
        "server/CommandManager.java",
        "server/GrpcServerImpl.java",
        "server/IdleServerTasks.java",
        "server/ServerWatcherRunnable.java",
    ],
    deps = [
        ":bug-report",
        ":runtime",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:command_server_java_grpc",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc:grpc-jar",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "bug-report",
    srcs = glob(["bugreport/*.java"]),
    deps = [
        ":blaze-version-info",
        "//src/main/java/com/google/devtools/build/lib/util:custom_exit_code_publisher",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:logging",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "runtime",
    srcs = glob(
        [
            "runtime/*.java",
            "runtime/commands/*.java",
            "runtime/mobileinstall/*.java",
            "buildtool/*.java",
            "buildtool/buildevent/*.java",
        ],
        exclude = [
            "buildtool/BuildRequestOptions.java",
            "runtime/KeepGoingOption.java",
            "runtime/LoadingPhaseThreadsOption.java",
        ],
    ) + [
        "server/RPCServer.java",
        "server/signal/InterruptSignalHandler.java",
    ],
    resources = glob([
        "runtime/commands/*.txt",
        "runtime/mobileinstall/*.txt",
    ]),
    deps = [
        ":bug-report",
        ":build-base",
        ":build-request-options",
        ":keep-going-option",
        ":loading-phase-threads-option",
        ":shared-base-rules",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/exec/local:options",
        "//src/main/java/com/google/devtools/build/lib/metrics:event",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/packages:build_type",
        "//src/main/java/com/google/devtools/build/lib/packages:starlark_semantics_options",
        "//src/main/java/com/google/devtools/build/lib/packages:type",
        "//src/main/java/com/google/devtools/build/lib/pkgcache",
        "//src/main/java/com/google/devtools/build/lib/platform:suspend_counter",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/profiler:profiler-output",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:allocationtracker",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2/common:abstract-blaze-query-env",
        "//src/main/java/com/google/devtools/build/lib/query2/engine",
        "//src/main/java/com/google/devtools/build/lib/query2/query/output",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/syntax:evaluator",
        "//src/main/java/com/google/devtools/build/lib/syntax:frontend",
        "//src/main/java/com/google/devtools/build/lib/unix",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:command",
        "//src/main/java/com/google/devtools/build/lib/util:custom_exit_code_publisher",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:filetype",
        "//src/main/java/com/google/devtools/build/lib/util:os",
        "//src/main/java/com/google/devtools/build/lib/util:process",
        "//src/main/java/com/google/devtools/build/lib/util:shell_escaper",
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:output_service",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:action_cache_java_proto",
        "//src/main/protobuf:analysis_java_proto",
        "//src/main/protobuf:bazel_flags_java_proto",
        "//src/main/protobuf:build_java_proto",
        "//src/main/protobuf:command_line_java_proto",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:option_filters_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "blaze-version-info",
    srcs = [
        "analysis/BlazeVersionInfo.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:string",
        "//third_party:guava",
    ],
)

java_library(
    name = "rule-definition-context",
    srcs = [
        "analysis/RuleDefinitionContext.java",
    ],
)

merge_licenses(
    name = "merge_licenses",
    srcs = [
        "//:LICENSE",
        "//third_party:srcs",
    ],
    out = "runtime/commands/LICENSE",
)

java_library(
    name = "all-build-rules",
    exports = [
        ":build-base",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "build",
    exports = [
        ":all-build-rules",
        ":build-base",
        ":foundation",
        ":query2",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/packages",
    ],
)

########################################################################
#
# The Blaze query language, revision 2: query2.
#

java_library(
    name = "query2",
    exports = [
        ":foundation",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_java_proto",
        "//third_party:guava",
    ],
)

filegroup(
    name = "docs_embedded_in_sources",
    srcs = glob(["**/*.java"]) + [
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genrule:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/objc:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/proto:srcs",
    ],
)

genrule(
    name = "gen_buildencyclopedia",
    srcs = [":docs_embedded_in_sources"],
    outs = ["build-encyclopedia.zip"],
    cmd = (
        "mkdir -p $(@D)/be && " +
        "$(location //src/main/java/com/google/devtools/build/docgen:docgen_bin)" +
        " --product_name=bazel" +
        " --provider=com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider" +
        " --input_dir=$$PWD/src/main/java/com/google/devtools/build/lib" +
        " --output_dir=$(@D)/be &&" +
        "zip -qj $@ $(@D)/be/*"
    ),
    tools = [
        "//src/main/java/com/google/devtools/build/docgen:docgen_bin",
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
    ],
    visibility = [
        "//site:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

genrule(
    name = "gen_command-line-reference",
    srcs = [
        "//site:command-line-reference-prefix.html",
        "//site:command-line-reference-suffix.html",
    ],
    outs = ["command-line-reference.html"],
    cmd = (
        "cat $(location //site:command-line-reference-prefix.html) > $@ && " +
        "TMP=`mktemp -d /tmp/tmp.XXXXXXXXXX` && " +
        "$(location //src/main/java/com/google/devtools/build/lib/bazel:BazelServer) " +
        "--jvm_flag=-Dio.bazel.EnableJni=0 --batch " +
        "--install_base=$${TMP} --output_base=$${TMP}/output/ --output_user_root=$${TMP} " +
        "help everything-as-html >> $@ 2>/dev/null && " +
        "cat $(location //site:command-line-reference-suffix.html) >> $@"
    ),
    tools = [
        "//src/main/java/com/google/devtools/build/lib/bazel:BazelServer",
    ],
    visibility = [
        "//site:__pkg__",
    ],
)

genrule(
    name = "gen_skylarklibrary",
    outs = ["skylark-library.zip"],
    cmd = "mkdir -p $(@D)/skylark-lib &&" +
          "$(location //src/main/java/com/google/devtools/build/docgen:skydoc_bin) $(@D)/skylark-lib" +
          " && zip -qj $@ $(@D)/skylark-lib/*",
    tools = ["//src/main/java/com/google/devtools/build/docgen:skydoc_bin"],
    visibility = ["//site:__pkg__"],
)

########################################################################
#
# Miscellaneous targets
#

# A library for just the BuildInfo class, so it can be used by the external
# SingleJar implementation.

java_library(
    name = "build-info",
    srcs = [
        "analysis/BuildInfo.java",
    ],
)
