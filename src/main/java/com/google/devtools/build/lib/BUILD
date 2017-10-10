# Description:
#   Main Java code for Bazel
package(
    default_visibility = ["//src:__subpackages__"],
)

# Generate list of all srcs via:
# blaze query 'filter("srcs", kind("filegroup rule", //third_party/bazel/src/main/java/com/google/devtools/build/lib/...))' | sort | sed -e "s/^/\"/" | sed -e "s/$/\",/" | fgrep -v "build/lib:srcs" | sed "s:third_party/bazel/::"
filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/build/docgen:srcs",
        "//src/main/java/com/google/devtools/build/lib/actions:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/analysis/whitelisting:srcs",
        "//src/main/java/com/google/devtools/build/lib/authandtls:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream:srcs",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports:srcs",
        "//src/main/java/com/google/devtools/build/lib/causes:srcs",
        "//src/main/java/com/google/devtools/build/lib/clock:srcs",
        "//src/main/java/com/google/devtools/build/lib/cmdline:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset:srcs",
        "//src/main/java/com/google/devtools/build/lib/collect:srcs",
        "//src/main/java/com/google/devtools/build/lib/concurrent:srcs",
        "//src/main/java/com/google/devtools/build/lib/exec/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/exec/local:srcs",
        "//src/main/java/com/google/devtools/build/lib/graph:srcs",
        "//src/main/java/com/google/devtools/build/lib/profiler:srcs",
        "//src/main/java/com/google/devtools/build/lib/query2:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/swift:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp/proto:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/genrule:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/objc:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/platform:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/lib/shell:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/packages:srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization:srcs",
        "//src/main/java/com/google/devtools/build/lib/ssd:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/vfs:srcs",
        "//src/main/java/com/google/devtools/build/lib/windows/jni:srcs",
        "//src/main/java/com/google/devtools/build/lib/windows/runfiles:srcs",
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

#   Java interfaces to various native UNIX system services.
java_library(
    name = "unix",
    srcs = glob([
        "unix/*.java",
    ]) + ["UnixJniLoader.java"],
    resources = select({
        # todo(dslomov): Get rid of this whole library on Windows.
        "//src:windows": [],
        "//src:windows_msys": [],
        "//src:windows_msvc": [],
        "//conditions:default": ["//src/main/native:libunix.so"],
    }),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
)

# String canonicalizer.
java_library(
    name = "base-util",
    srcs = [
        "util/StringCanonicalizer.java",
        "util/VarInt.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//third_party:guava",
    ],
)

# Varargs-avoiding overloads of Preconditions functions.
java_library(
    name = "preconditions",
    srcs = ["util/Preconditions.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# I/O utilities.
java_library(
    name = "io",
    srcs = glob(["util/io/*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
    ],
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
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//third_party:guava",
    ],
)

java_library(
    name = "process_util",
    srcs = ["util/ProcessUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/windows",
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
    name = "util",
    srcs = glob(
        ["util/*.java"],
        exclude = [
            "util/BlazeClock.java",
            "util/Clock.java",
            "util/ExitCode.java",
            "util/JavaClock.java",
            "util/OS.java",
            "util/Preconditions.java",
            "util/ProcessUtils.java",
            "util/SingleLineFormatter.java",
            "util/StringCanonicalizer.java",
            "util/StringTrie.java",
            "util/VarInt.java",
        ],
    ),
    exports = [
        "//src/main/java/com/google/devtools/build/lib:base-util",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/collect",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:base-util",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/compacthashset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
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

# Event reporting infrastructure.
java_library(
    name = "events",
    srcs = glob(["events/*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
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
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
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
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
    ],
)

java_library(
    name = "skylarkinterface",
    srcs = glob([
        "skylarkinterface/*.java",
    ]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:util",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "syntax",
    srcs = glob([
        "syntax/**/*.java",
    ]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-util",
        "//third_party:auto_value",
        "//third_party:bytebuddy",
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
    srcs = glob([
        "packages/*.java",
        "pkgcache/*.java",
    ]),
    exports = [
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:config-matching-provider",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib:transitive-info-provider",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "packages",
    exports = [
        "//src/main/java/com/google/devtools/build/lib:foundation",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
    ],
)

java_library(
    name = "java-toolchain-parser",
    srcs = [
        "rules/java/JavaToolchainData.java",
        "rules/java/JavaToolchainDataParser.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:preconditions",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/protobuf:build_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
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
        "//third_party:guava",
    ],
)

java_library(
    name = "skylark-provider-collection",
    srcs = ["analysis/SkylarkProviderCollection.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
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
        "//src/main/java/com/google/devtools/build/lib:build-base",
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
            "rules/test/*.java",
            "rules/repository/*.java",
            "skyframe/*.java",
        ],
        exclude = [
            "analysis/BuildInfo.java",
            "analysis/TransitiveInfoProvider.java",
            "analysis/config/BuildConfigurationOptionDetails.java",
        ],
    ) + [
        "runtime/BlazeServerStartupOptions.java",
        "runtime/BuildEventStreamerUtils.java",
    ],
    exports = [
        "//src/main/java/com/google/devtools/build/lib:transitive-info-provider",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:base-util",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:os_util",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:skylark-provider-collection",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:syntax",
        "//src/main/java/com/google/devtools/build/lib:transitive-info-provider",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:utils",
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
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
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
        "//src/main/java/com/google/devtools/build/lib:RpcSupport",
        "//src/main/java/com/google/devtools/build/lib:android-rules",
        "//src/main/java/com/google/devtools/build/lib:bazel",
        "//src/main/java/com/google/devtools/build/lib:bazel-repository",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:build-info",
        "//src/main/java/com/google/devtools/build/lib:core-rules",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:java-compilation",
        "//src/main/java/com/google/devtools/build/lib:java-rules",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/query2:query-output",
        "//src/main/java/com/google/devtools/build/lib/rules/apple",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/swift",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp/proto:CcProtoLibrary",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery",
        "//src/main/java/com/google/devtools/build/lib/rules/genrule",
        "//src/main/java/com/google/devtools/build/lib/rules/objc",
        "//src/main/java/com/google/devtools/build/lib/rules/platform",
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
    name = "bazel-main",
    srcs = glob(["bazel/*.java"]),
    resources = [
        "bazel/rules/java/java_stub_template.txt",
        "bazel/rules/java/java_stub_template_windows.txt",
        "bazel/rules/python/python_stub_template.txt",
        "bazel/rules/sh/sh_stub_template_windows.txt",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:bazel",
        "//src/main/java/com/google/devtools/build/lib:bazel-commands",
        "//src/main/java/com/google/devtools/build/lib:bazel-repository",
        "//src/main/java/com/google/devtools/build/lib:bazel-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:build-info",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/remote",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/ssd",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "bazel-commands",
    srcs = glob(["bazel/commands/*.java"]),
    resources = ["bazel/commands/fetch.txt"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//src/main/java/com/google/devtools/build/lib:java-compilation",
        "//src/main/java/com/google/devtools/build/lib:java-rules",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2:query-engine",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
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
    runtime_deps = [
        "//third_party:aether",
        "//third_party:apache_commons_codec",
        "//third_party:apache_commons_logging",
        "//third_party:apache_httpclient",
        "//third_party:apache_httpcore",
        "//third_party:jsch",
        "//third_party:maven",
        "//third_party:maven_model",
        "//third_party:plexus_interpolation",
        "//third_party:plexus_utils",
        "//third_party:slf4j",
    ],
    deps = [
        "//src/java_tools/singlejar/java/com/google/devtools/build/zip",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:maven-connector",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:aether",
        "//third_party:apache_commons_compress",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jgit",
        "//third_party:jsr305",
        "//third_party:maven",
        "//third_party:maven_model",
        "//third_party:plexus_component_annotations",
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
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/vfs",
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
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "java-rules",
    srcs = [
        "rules/java/GenericBuildInfoPropertiesTranslator.java",
        "rules/java/JavaBinary.java",
        "rules/java/JavaCcLinkParamsProvider.java",
        "rules/java/JavaConfigurationLoader.java",
        "rules/java/JavaImport.java",
        "rules/java/JavaImportBaseRule.java",
        "rules/java/JavaInfo.java",
        "rules/java/JavaLibrary.java",
        "rules/java/JavaPlugin.java",
        "rules/java/JavaPrimaryClassProvider.java",
        "rules/java/JavaRuntime.java",
        "rules/java/JavaRuntimeAlias.java",
        "rules/java/JavaRuntimeClasspathProvider.java",
        "rules/java/JavaRuntimeJarProvider.java",
        "rules/java/JavaRuntimeRule.java",
        "rules/java/JavaRuntimeSuite.java",
        "rules/java/JavaRuntimeSuiteRule.java",
        "rules/java/JavaSkylarkCommon.java",
        "rules/java/JavaSourceInfoProvider.java",
        "rules/java/JavaToolchain.java",
        "rules/java/JavaToolchainAlias.java",
        "rules/java/JavaToolchainRule.java",
        "rules/java/JavaToolchainSkylarkApiProvider.java",
        "rules/java/JvmConfigurationLoader.java",
        "rules/java/OneVersionCheckActionBuilder.java",
        "rules/java/ProguardHelper.java",
        "rules/java/ProguardLibrary.java",
        "rules/java/ProguardLibraryRule.java",
        "rules/java/ProguardSpecProvider.java",
        "rules/java/ProtoJavaApiInfoAspectProvider.java",
        "rules/java/ProtoJavaApiInfoProvider.java",
        "rules/java/proto/ActionReuser.java",
        "rules/java/proto/JavaLiteProtoAspect.java",
        "rules/java/proto/JavaLiteProtoLibrary.java",
        "rules/java/proto/JavaProtoAspect.java",
        "rules/java/proto/JavaProtoLibrary.java",
        "rules/java/proto/JavaProtoLibraryAspectProvider.java",
        "rules/java/proto/JavaProtoSkylarkCommon.java",
        "rules/java/proto/JplCcLinkParams.java",
        "rules/java/proto/ProtoJavacOpts.java",
        "rules/java/proto/StrictDepsUtils.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:RpcSupport",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:java-compilation",
        "//src/main/java/com/google/devtools/build/lib:java-implicit-attributes",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "RpcSupport",
    srcs = ["rules/java/proto/RpcSupport.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:java-compilation",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//third_party:guava",
    ],
)

java_library(
    name = "java-compilation",
    srcs = [
        "rules/java/BuildInfoPropertiesTranslator.java",
        "rules/java/ClasspathConfiguredFragment.java",
        "rules/java/DeployArchiveBuilder.java",
        "rules/java/JavaBuildInfoFactory.java",
        "rules/java/JavaCommon.java",
        "rules/java/JavaCompilationArgs.java",
        "rules/java/JavaCompilationArgsProvider.java",
        "rules/java/JavaCompilationArtifacts.java",
        "rules/java/JavaCompilationHelper.java",
        "rules/java/JavaCompilationInfoProvider.java",
        "rules/java/JavaCompileAction.java",
        "rules/java/JavaConfiguration.java",
        "rules/java/JavaConstraintProvider.java",
        "rules/java/JavaExportsProvider.java",
        "rules/java/JavaGenJarsProvider.java",
        "rules/java/JavaHeaderCompileAction.java",
        "rules/java/JavaHelper.java",
        "rules/java/JavaInfo.java",
        "rules/java/JavaLibraryHelper.java",
        "rules/java/JavaNativeLibraryProvider.java",
        "rules/java/JavaNeverlinkInfoProvider.java",
        "rules/java/JavaOptions.java",
        "rules/java/JavaPluginInfoProvider.java",
        "rules/java/JavaRuleOutputJarsProvider.java",
        "rules/java/JavaRunfilesProvider.java",
        "rules/java/JavaRuntimeInfo.java",
        "rules/java/JavaSemantics.java",
        "rules/java/JavaSkylarkApiProvider.java",
        "rules/java/JavaSourceJarsProvider.java",
        "rules/java/JavaTargetAttributes.java",
        "rules/java/JavaToolchainData.java",
        "rules/java/JavaToolchainProvider.java",
        "rules/java/JavaUtil.java",
        "rules/java/Jvm.java",
        "rules/java/MessageBundleProvider.java",
        "rules/java/NativeLibraryNestedSetBuilder.java",
        "rules/java/ProtoJavaApiInfoAspectProvider.java",
        "rules/java/ProtoJavaApiInfoProvider.java",
        "rules/java/ResourceJarActionBuilder.java",
        "rules/java/SingleJarActionBuilder.java",
        "rules/java/WriteBuildInfoPropertiesAction.java",
        "rules/java/proto/GeneratedExtensionRegistryProvider.java",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:build-info",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:java-implicit-attributes",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "java-implicit-attributes",
    srcs = ["rules/java/JavaImplicitAttributes.java"],
)

java_library(
    name = "core-rules",
    srcs = glob(
        ["rules/core/*.java"],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//third_party:guava",
    ],
)

java_library(
    name = "android-rules",
    srcs = glob(
        ["rules/android/*.java"],
    ),
    resources = [
        "rules/android/android_device_stub_template.txt",
        "rules/android/android_instrumentation_test_template.txt",
        "rules/android/databinding_annotation_template.txt",
        "rules/android/test_suite_property_name.txt",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:java-compilation",
        "//src/main/java/com/google/devtools/build/lib:java-rules",
        "//src/main/java/com/google/devtools/build/lib:nativedeps-rules",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis/whitelisting",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/config",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:android_deploy_info_java_proto",
        "//src/main/protobuf:apk_manifest_java_proto",
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
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:skylarkinterface",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:crosstool_config_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/protobuf:protobuf_java",
    ],
)

java_library(
    name = "shared-base-rules",
    exports = [
        "//src/main/java/com/google/devtools/build/lib:android-rules",
        "//src/main/java/com/google/devtools/build/lib:java-compilation",
        "//src/main/java/com/google/devtools/build/lib:java-rules",
        "//src/main/java/com/google/devtools/build/lib:nativedeps-rules",
        "//src/main/java/com/google/devtools/build/lib:proto-rules",
        "//src/main/java/com/google/devtools/build/lib:python-rules",
    ],
)

java_library(
    name = "runtime",
    srcs = glob(
        [
            "runtime/*.java",
            "server/**/*.java",
            "runtime/commands/*.java",
            "runtime/mobileinstall/*.java",
            "buildtool/*.java",
            "buildtool/buildevent/*.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:exitcode-external",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:process_util",
        "//src/main/java/com/google/devtools/build/lib:shared-base-rules",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/exec/local",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:profiler-output",
        "//src/main/java/com/google/devtools/build/lib/query2",
        "//src/main/java/com/google/devtools/build/lib/query2:query-engine",
        "//src/main/java/com/google/devtools/build/lib/query2:query-output",
        "//src/main/java/com/google/devtools/build/lib/shell",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/windows",
        "//src/main/java/com/google/devtools/build/skyframe",
        "//src/main/java/com/google/devtools/build/skyframe:skyframe-objects",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/java/com/google/devtools/common/options:invocation_policy",
        "//src/main/protobuf:action_cache_java_proto",
        "//src/main/protobuf:bazel_flags_java_proto",
        "//src/main/protobuf:build_java_proto",
        "//src/main/protobuf:command_line_java_proto",
        "//src/main/protobuf:command_server_java_grpc",
        "//src/main/protobuf:command_server_java_proto",
        "//src/main/protobuf:extra_actions_base_java_proto",
        "//src/main/protobuf:invocation_policy_java_proto",
        "//src/main/protobuf:option_filters_java_proto",
        "//src/main/protobuf:test_status_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc:grpc-jar",
        "//third_party/protobuf:protobuf_java",
    ],
)

genrule(
    name = "merge_licenses",
    srcs = [
        "//:LICENSE",
        "//third_party:srcs",
    ],
    outs = ["runtime/commands/LICENSE"],
    # Use xargs to avoid the shell script complaining that is has been passed too many arguments.
    # xargs calls the script multiple times with at most 1000 arguments per call.
    cmd = "echo \"$(SRCS)\" | xargs -n1000 $(location merge_licenses.sh) > \"$@\"",
    tools = ["merge_licenses.sh"],
)

java_library(
    name = "bazel",
    resources = glob([
        "runtime/commands/*.txt",
        "runtime/mobileinstall/*.txt",
    ]) + [
        "//src/main/java/com/google/devtools/build/lib:runtime/commands/LICENSE",
    ],
    exports = ["//src/main/java/com/google/devtools/build/lib:runtime"],
)

java_library(
    name = "all-build-rules",
    exports = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:util",
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
        "//src/main/java/com/google/devtools/build/lib:all-build-rules",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:foundation",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:query2",
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
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:foundation",
        "//src/main/java/com/google/devtools/build/lib:packages",
        "//src/main/java/com/google/devtools/build/lib:util",
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
    main_class = "com.google.devtools.build.lib.bazel.BazelMain",
    runtime_deps = [
        "//src/main/java/com/google/devtools/build/lib:bazel-main",
        "//src/main/java/com/google/devtools/build/lib:single-line-formatter",  # See startup_options.cc
    ],
)

filegroup(
    name = "docs_embedded_in_sources",
    srcs = glob(["**/*.java"]) + [
        "//src/main/java/com/google/devtools/build/lib/rules/apple:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/apple/cpp:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/config:srcs",
        "//src/main/java/com/google/devtools/build/lib/rules/cpp:srcs",
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
    srcs = ["//src/main/java/com/google/devtools/build/lib:docs_embedded_in_sources"],
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
        "--install_base=$${TMP} --output_base=$${TMP}/output/ " +
        "help everything-as-html >> $@ 2>/dev/null && " +
        "cat $(location //site:command-line-reference-suffix.html) >> $@"
    ),
    tools = [
        "//src/main/java/com/google/devtools/build/lib:bazel/BazelServer",
    ],
    visibility = [
        "//site:__pkg__",
    ],
)

# The skylark repository classes are passed as parameter of the Skylark documentation generator.
SKYLARK_REPOSITORY_CLASSES = [
    "com.google.devtools.build.lib.bazel.repository.skylark.SkylarkExecutionResult",
    "com.google.devtools.build.lib.bazel.repository.skylark.SkylarkOS",
    "com.google.devtools.build.lib.bazel.repository.skylark.SkylarkPath",
    "com.google.devtools.build.lib.bazel.repository.skylark.SkylarkRepositoryContext",
    "com.google.devtools.build.lib.bazel.repository.skylark.SkylarkRepositoryModule",
]

genrule(
    name = "gen_skylarklibrary",
    outs = ["skylark-library.zip"],
    cmd = "mkdir -p $(@D)/skylark-lib &&" +
          "$(location //src/main/java/com/google/devtools/build/docgen:skydoc_bin) $(@D)/skylark-lib external" +
          " ".join(SKYLARK_REPOSITORY_CLASSES) +
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
