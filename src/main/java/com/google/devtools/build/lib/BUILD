# Description:
#   Main Java code for Bazel
package(
    default_visibility = ["//src:__subpackages__"],
)

load(":merge_licenses.bzl", "merge_licenses")

# Generate list of all srcs via:
# bazel query 'filter("srcs", kind("filegroup rule", //src/main/java/com/google/devtools/build/lib/...))' | sort | sed -e "s/^/\"/" | sed -e "s/$/\",/" | fgrep -v "build/lib:srcs"
filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/docgen:srcs",
        "//src/main/java/com/google/devtools/build/lib/actions:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:srcs",
        "//src/main/java/com/google/devtools/build/lib/authandtls:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports:srcs",
        "//src/main/java/com/google/devtools/build/lib/causes:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/debug:srcs",
        "//src/main/java/com/google/devtools/build/lib/clock:srcs",
        "//src/main/java/com/google/devtools/build/lib/cmdline:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect:srcs",
        "//src/main/java/com/google/devtools/build/lib/concurrent:srcs",
        "//src/main/java/com/google/devtools/build/lib/dynamic:srcs",
        "//src/main/java/com/google/devtools/build/lib/exec/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/exec/local:srcs",
        "//src/main/java/com/google/devtools/build/lib/graph:srcs",
        "//src/main/java/com/google/devtools/build/lib/metrics:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler/callcounts:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/blobstore:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/blobstore/http:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/logging:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/util:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/swift:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genrule:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/objc:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/lib/shell:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/module:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/server:srcs",
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface/processor:srcs",
        "//src/main/java/com/google/devtools/build/lib/ssd:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/windows/jni:srcs",
        "//src/main/java/com/google/devtools/build/lib/windows:srcs",
        "//src/main/java/com/google/devtools/build/lib/worker:srcs",
        "//src/main/java/com/google/devtools/build/lib/unsafe:srcs",
        "//src/main/java/com/google/devtools/build/skyframe:srcs",
        "//src/main/java/com/google/devtools/common/options:srcs",
    ],
    visibility = [
        "//src:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

#   Java interfaces to various native UNIX system services.
java_library(
    name = "unix",
    srcs = glob([
        "unix/*.java",
    ]) + ["UnixJniLoader.java"],
    resources = select({
        # todo(dslomov): Get rid of this whole library on Windows.
        "//src/conditions:windows": [],
        "//conditions:default": ["//src/main/native:libunix.so"],
    }),
    deps = [
        ":os_util",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:flogger",
        "//third_party:guava",
    ],
)

# String canonicalizer.
java_library(
    name = "base-util",
    srcs = [
        "util/VarInt.java",
    ],
    exports = [":string_util"],
)

# I/O utilities.

OUT_ERR_SRCS = [
    "util/io/OutErr.java",
    "util/io/LinePrefixingOutputStream.java",
    "util/io/LineFlushingOutputStream.java",
]

java_library(
    name = "io",
    srcs = glob(
        ["util/io/*.java"],
        exclude = OUT_ERR_SRCS,
    ),
    deps = [
        ":out-err",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "out-err",
    srcs = OUT_ERR_SRCS,
)

# General utilities.
java_library(
    name = "os_util",
    srcs = ["util/OS.java"],
)

java_library(
    name = "classpath-util",
    srcs = ["util/Classpath.java"],
    deps = [
        "//third_party:guava",
    ],
)

java_library(
    name = "process_util",
    srcs = ["util/ProcessUtils.java"],
    deps = [
        ":os_util",
        ":unix",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/windows/jni:processes",
    ],
)

java_library(
    name = "single-line-formatter",
    srcs = ["util/SingleLineFormatter.java"],
    deps = [
        "//third_party:guava",
    ],
)

java_library(
    name = "simple-log-handler",
    srcs = ["util/SimpleLogHandler.java"],
    deps = [
        ":os_util",
        ":single-line-formatter",
        ":util",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

RESOURCE_UTILS_SRCS = [
    "util/ResourceConverter.java",
]

COMMAND_UTILS_SRCS = [
    "util/CommandBuilder.java",
    "util/CommandDescriptionForm.java",
    "util/CommandFailureUtils.java",
    "util/CommandUtils.java",
]

java_library(
    name = "command-utils",
    srcs = COMMAND_UTILS_SRCS,
    deps = [
        ":util",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "util",
    srcs = glob(
        ["util/*.java"],
        exclude = COMMAND_UTILS_SRCS + RESOURCE_UTILS_SRCS + [
            "util/BlazeClock.java",
            "util/Clock.java",
            "util/ExitCode.java",
            "util/FileType.java",
            "util/FileTypeSet.java",
            "util/JavaClock.java",
            "util/OS.java",
            "util/ProcessUtils.java",
            "util/ResourceUsage.java",
            "util/SimpleLogHandler.java",
            "util/SingleLineFormatter.java",
            "util/StringCanonicalizer.java",
            "util/StringTrie.java",
            "util/StringUtil.java",
            "util/StringUtilities.java",
            "util/VarInt.java",
        ],
    ),
    exports = [
        ":base-util",
        ":exitcode-external",
        ":filetype",
        ":os_util",
        ":resource_usage",
        ":string_util",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect",
    ],
    deps = [
        ":base-util",
        ":exitcode-external",
        ":os_util",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:apache_commons_lang",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "filetype",
    srcs = [
        "util/FileType.java",
        "util/FileTypeSet.java",
    ],
    deps = [
        ":string_util",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "resource_usage",
    srcs = [
        "util/ResourceUsage.java",
    ],
    deps = [
        "//third_party:guava",
    ],
)

java_library(
    name = "string_util",
    srcs = [
        "util/StringCanonicalizer.java",
        "util/StringUtil.java",
        "util/StringUtilities.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
    ],
)

java_library(
    name = "exitcode-external",
    srcs = [
        "util/ExitCode.java",
    ],
    deps = [
        "//third_party:guava",
    ],
)

java_library(
    name = "custom-exit-code-publisher",
    srcs = ["util/CustomExitCodePublisher.java"],
    deps = [
        ":exitcode-external",
        "//third_party:jsr305",
    ],
)

# Event reporting infrastructure.
java_library(
    name = "events",
    srcs = glob(["events/*.java"]),
    deps = [
        ":io",
        ":out-err",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party:jsr305",
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
        ":events",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
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
        ":events",
        ":skylarkinterface",
        ":syntax",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
    ],
)

java_library(
    name = "skylarkinterface",
    exported_plugins = ["//src/main/java/com/google/devtools/build/lib/skylarkinterface/processor:annotation_preprocessor"],
    exports = [":skylarkinterface_internal"],
)

java_library(
    name = "skylarkinterface_internal",
    srcs = glob([
        "skylarkinterface/*.java",
    ]),
    visibility = [
        "//src/main/java/com/google/devtools/build/lib/skylarkinterface/processor:__pkg__",
        "//src/test/java/com/google/devtools/build/lib/skyframe:__pkg__",
    ],
    deps = [
        ":skylark_semantics",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "starlark_context",
    srcs = [
        "analysis/skylark/BazelStarlarkContext.java",
        "analysis/skylark/SymbolGenerator.java",
    ],
    deps = [
        ":skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/cmdline:RepositoryName",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "skylark_semantics",
    srcs = [
        "syntax/SkylarkSemantics.java",
    ],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "skylark_semantics_options",
    srcs = [
        "packages/SkylarkSemanticsOptions.java",
    ],
    deps = [
        ":skylark_semantics",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
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
        ":util",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "skylark-semantics",
    srcs = ["syntax/SkylarkSemantics.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

java_library(
    name = "syntax",
    srcs = glob(
        [
            "syntax/**/*.java",
        ],
        exclude = ["syntax/SkylarkSemantics.java"],
    ),
    deps = [
        ":events",
        ":skylark-semantics",
        ":skylarkinterface",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

########################################################################
#
# The "packages" library: syntax, packages, pkgcache.
# BUILD file scanning, parsing and evaluation; package loading and caching.
#

# IMPORTANT:  NOT A PUBLIC INTERFACE.
#
# DO NOT ADD A DEPENDENCY ON THIS TARGET, or any other in this
# package, without first talking to us.  Here are two good
# reasons:
#
# 1.  FRAGILITY. This is not a stable API, it's an internal interface.
#     It changes from time to time, and your client *will* be broken.
#
# 2.  CORRECTNESS.  The BUILD file parser changes rapidly.  If
#     your application depends upon it, your application will start to
#     bit-rot almost immediately.

java_library(
    name = "packages-internal",
    srcs = glob(
        [
            "packages/*.java",
            "pkgcache/*.java",
        ],
        exclude = ["packages/BuilderFactoryForTesting.java"],
    ),
    exports = [
        ":syntax",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
    ],
    deps = [
        ":config-matching-provider",
        ":config-transitions",
        ":events",
        ":skylark_semantics",
        ":skylarkinterface",
        ":starlark_context",
        ":syntax",
        ":transitive-info-provider",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "packages/BuilderFactoryForTesting",
    srcs = ["packages/BuilderFactoryForTesting.java"],
    deps = [
        ":build-base",
        ":packages-internal",
        "//third_party:guava",
    ],
)

java_library(
    name = "packages",
    exports = [
        ":foundation",
        ":packages-internal",
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
        ":packages-internal",
        ":transitive-info-provider",
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
        ],
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
        ":transitive-info-provider",
    ],
    deps = [
        ":bug-report",
        ":build-request-options",
        ":command-utils",
        ":events",
        ":exitcode-external",
        ":io",
        ":keep-going-option",
        ":os_util",
        ":out-err",
        ":packages-internal",
        ":process_util",
        ":provider-collection",
        ":resource-converter",
        ":skylark_semantics",
        ":skylarkinterface",
        ":syntax",
        ":transitive-info-provider",
        ":unix",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/causes",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:current_rule_tracker",
        "//src/main/java/com/google/devtools/build/lib/query2:aquery-utils",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:cpp_interface",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skyframe/trimming:trimmed_configuration_cache",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/config",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/platform",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:output_service",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:analysis_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:spawn_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "bazel-rules",
    srcs = glob(
        [
            "bazel/rules/**/*.java",
        ],
        exclude = [
            "bazel/rules/workspace/*.java",
        ],
    ),
    resources = glob(
        [
            "bazel/rules/**/*.txt",
            "bazel/rules/**/*.WORKSPACE",
        ],
    ),
    deps = [
        ":android-rules",
        ":bazel",
        ":bazel-repository",
        ":build-base",
        ":build-info",
        ":core-rules",
        ":core-workspace-rules",
        ":events",
        ":packages-internal",
        ":proto-rules",
        ":python-rules",
        ":testing-support-rules",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/swift",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp/proto:CcProtoLibrary",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery",
        "//src/main/java/com/google/devtools/build/lib/rules/genrule",
        "//src/main/java/com/google/devtools/build/lib/rules/java:RpcSupport",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/rules/platform",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/android",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/apple",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/cpp",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/java",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/proto",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/python",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//third_party:android_common_25_0_0",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "bazel/BazelRepositoryModule",
    srcs = ["bazel/BazelRepositoryModule.java"],
    deps = [
        ":bazel-commands",
        ":bazel-repository",
        ":bazel-rules",
        ":build-base",
        ":events",
        ":io",
        ":packages-internal",
        ":runtime",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/repository",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options:options_internal",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "bazel-main",
    srcs = ["bazel/Bazel.java"],
    resources = [
        "bazel/rules/java/java_stub_template.txt",
        "bazel/rules/python/python_stub_template.txt",
    ],
    deps = [
        ":bazel",
        ":bazel-coverage",
        ":bazel-modules",
        ":bazel-repository",
        ":bazel-rules",
        ":bazel/BazelRepositoryModule",
        ":build-base",
        "//src/main/java/com/google/devtools/build/lib/bazel/debug:workspace-rule-module",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice",
        "//src/main/java/com/google/devtools/build/lib/dynamic",
        "//src/main/java/com/google/devtools/build/lib/metrics:metrics_module",
        "//src/main/java/com/google/devtools/build/lib/profiler/callcounts:callcounts_module",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:allocationtracker_module",
        "//src/main/java/com/google/devtools/build/lib/remote",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/skylarkdebug/module",
        "//src/main/java/com/google/devtools/build/lib/ssd",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//third_party:guava",
    ],
)

java_library(
    name = "bazel-modules",
    srcs = glob(
        ["bazel/*.java"],
        exclude = [
            "bazel/Bazel.java",
            "bazel/BazelRepositoryModule.java",
        ],
    ),
    deps = [
        ":build-base",
        ":build-info",
        ":command-utils",
        ":events",
        ":exitcode-external",
        ":io",
        ":runtime",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "bazel-commands",
    srcs = glob(["bazel/commands/*.java"]),
    resources = [
        "bazel/commands/fetch.txt",
        "bazel/commands/sync.txt",
    ],
    deps = [
        ":bazel-repository",
        ":build-base",
        ":events",
        ":exitcode-external",
        ":keep-going-option",
        ":loading-phase-threads-option",
        ":packages-internal",
        ":runtime",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/query2:abstract-blaze-query-env",
        "//src/main/java/com/google/devtools/build/lib/query2:query-engine",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
    ],
)

java_library(
    name = "bazel-coverage",
    srcs = glob(["bazel/coverage/*.java"]),
    deps = [
        ":build-base",
        ":events",
        ":runtime",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "bazel-repository",
    srcs = glob(
        [
            "bazel/repository/**/*.java",
            "bazel/rules/workspace/*.java",
        ],
        exclude = ["bazel/repository/MavenConnector.java"],
    ),
    resources = [
        "bazel/repository/local_config_platform.WORKSPACE",
    ],
    runtime_deps = [
        "//third_party:jsch",
        "//third_party:maven_model",
        "//third_party:plexus_component_annotations",
        "//third_party:plexus_interpolation",
        "//third_party:plexus_utils",
        "//third_party:slf4j",
    ],
    deps = [
        ":build-base",
        ":events",
        ":io",
        ":maven-connector",
        ":out-err",
        ":packages-internal",
        ":runtime",
        ":skylark_semantics",
        ":skylarkinterface",
        ":string_util",
        ":util",
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bazel/debug:workspace-rule-event",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/repository",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:aether",
        "//third_party:apache_commons_compress",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:maven",
        "//third_party:xz",
    ],
)

java_library(
    name = "maven-connector",
    srcs = ["bazel/repository/MavenConnector.java"],
    deps = [
        "//third_party:aether",
        "//third_party:maven_model",
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
        ":packages-internal",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "proto-rules",
    srcs = glob(
        ["rules/proto/*.java"],
    ),
    deps = [
        ":build-base",
        ":events",
        ":packages-internal",
        ":skylarkinterface",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/proto",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
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
        ":packages-internal",
        ":skylarkinterface",
        ":syntax",
        ":transitive-info-provider",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_proto",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/test",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:guava",
    ],
)

java_library(
    name = "android-rules",
    srcs = glob(
        [
            "rules/android/*.java",
            "rules/android/databinding/*.java",
        ],
    ),
    resources = [
        "rules/android/android_device_stub_template.txt",
        "rules/android/android_instrumentation_test_template.txt",
        "rules/android/databinding/databinding_annotation_template.txt",
        "rules/android/robolectric_properties_template.txt",
        "rules/android/test_suite_property_name.txt",
    ],
    deps = [
        ":build-base",
        ":events",
        ":nativedeps-rules",
        ":packages-internal",
        ":proto-rules",
        ":skylarkinterface",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:commandline_item",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/android",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:android_deploy_info_java_proto",
        "//third_party:android_common_25_0_0",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "python-rules",
    srcs = glob(
        ["rules/python/*.java"],
    ),
    deps = [
        ":build-base",
        ":events",
        ":filetype",
        ":os_util",
        ":packages-internal",
        ":skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec",
        "//src/main/java/com/google/devtools/build/lib/skylarkbuildapi/python",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "shared-base-rules",
    exports = [
        ":android-rules",
        ":nativedeps-rules",
        ":proto-rules",
        ":python-rules",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-compilation",
        "//src/main/java/com/google/devtools/build/lib/rules/java:java-rules",
    ],
)

java_library(
    name = "build-request-options",
    srcs = ["buildtool/BuildRequestOptions.java"],
    deps = [
        ":resource-converter",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
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
        ":resource-converter",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "resource-converter",
    srcs = RESOURCE_UTILS_SRCS,
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:apache_commons_lang",
        "//third_party:guava",
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
        ":exitcode-external",
        ":io",
        ":os_util",
        ":out-err",
        ":runtime",
        ":string_util",
        ":unix",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:command_server_java_grpc",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc:grpc-jar",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "bug-report",
    srcs = ["bugreport/BugReport.java"],
    deps = [
        ":blaze-version-info",
        ":custom-exit-code-publisher",
        ":exitcode-external",
        ":logging-util",
        ":out-err",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "logging-util",
    srcs = ["util/LoggingUtil.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
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
    deps = [
        ":bug-report",
        ":build-base",
        ":build-request-options",
        ":command-utils",
        ":custom-exit-code-publisher",
        ":events",
        ":exitcode-external",
        ":io",
        ":keep-going-option",
        ":loading-phase-threads-option",
        ":out-err",
        ":packages-internal",
        ":process_util",
        ":shared-base-rules",
        ":unix",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:localhost_capacity",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/exec/local:options",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:profiler-output",
        "//src/main/java/com/google/devtools/build/lib/profiler/memory:allocationtracker",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2:abstract-blaze-query-env",
        "//src/main/java/com/google/devtools/build/lib/query2:aquery-utils",
        "//src/main/java/com/google/devtools/build/lib/query2:query-engine",
        "//src/main/java/com/google/devtools/build/lib/query2:query-output",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:output_service",
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
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:option_filters_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:auto_value",
        "//third_party:flogger",
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
        ":string_util",
        "//third_party:guava",
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
    name = "bazel",
    resources = glob([
        "runtime/commands/*.txt",
        "runtime/mobileinstall/*.txt",
    ]) + [
        ":runtime/commands/LICENSE",
    ],
    exports = [":runtime"],
)

java_library(
    name = "all-build-rules",
    exports = [
        ":build-base",
        ":packages-internal",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
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
        ":packages",
        ":query2",
        "//src/main/java/com/google/devtools/build/lib/actions",
    ],
)

########################################################################
#
# The Blaze query language, revision 2: query2.
#

java_library(
    name = "query2",
    exports = [
        ":events",
        ":foundation",
        ":packages",
        ":util",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/graph",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_java_proto",
        "//third_party:guava",
    ],
)

java_binary(
    name = "bazel/BazelServer",
    javacopts = [
        "-Xlint:all",
        "-Xlint:-path",
    ],
    main_class = "com.google.devtools.build.lib.bazel.Bazel",
    runtime_deps = [
        ":bazel-main",
        ":server",
        ":single-line-formatter",  # See startup_options.cc
        "//src/main/java/com/google/devtools/build/lib:simple-log-handler",  # See startup_options.cc
    ],
)

filegroup(
    name = "docs_embedded_in_sources",
    srcs = glob(["**/*.java"]) + [
        "//src/main/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/java:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genrule:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/objc:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/platform:srcs",
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
        "mkdir $${TMP}/_embedded_binaries/ && " +
        "$(location :bazel/BazelServer) " +
        "--jvm_flag=-Dio.bazel.EnableJni=0 --batch " +
        "--install_base=$${TMP} --output_base=$${TMP}/output/ --output_user_root=$${TMP} " +
        "help everything-as-html >> $@ 2>/dev/null && " +
        "cat $(location //site:command-line-reference-suffix.html) >> $@"
    ),
    tools = [
        ":bazel/BazelServer",
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
