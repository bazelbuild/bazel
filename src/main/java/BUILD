package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "shell",
    srcs = glob([
        "com/google/devtools/build/lib/shell/*.java",
    ]),
    deps = ["//third_party:guava"],
)

java_library(
    name = "cmdline",
    srcs = glob([
        "com/google/devtools/build/lib/cmdline/*.java",
    ]),
    deps = [
        ":base-util",
        ":common",
        ":concurrent",
        ":syntax",
        ":util",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "collect",
    srcs = glob([
        "com/google/devtools/build/lib/collect/**/*.java",
    ]),
    deps = [
        ":common",
        ":shell",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "common",
    srcs = [
        "com/google/devtools/build/lib/Constants.java",
    ],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "concurrent",
    srcs = glob([
        "com/google/devtools/build/lib/concurrent/*.java",
    ]),
    deps = [
        ":common",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "events",
    srcs = glob([
        "com/google/devtools/build/lib/events/*.java",
    ]),
    deps = [
        ":common",
        ":concurrent",
        ":io",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "syntax",
    srcs = glob(["com/google/devtools/build/lib/syntax/*.java"]),
    deps = [
        ":base-util",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "packages",
    srcs = glob([
        "com/google/devtools/build/lib/packages/*.java",
        "com/google/devtools/build/lib/pkgcache/*.java",
    ]),
    exports = [
        ":cmdline",
        ":syntax",
    ],
    deps = [
        ":cmdline",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":syntax",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "unix",
    srcs = glob([
        "com/google/devtools/build/lib/unix/*.java",
    ]) + [
        "com/google/devtools/build/lib/UnixJniLoader.java",
    ],
    deps = [
        ":common",
        ":shell",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "vfs",
    srcs = glob([
        "com/google/devtools/build/lib/vfs/**/*.java",
        "com/google/devtools/build/lib/profiler/*.java",
    ]),
    deps = [
        ":base-util",
        ":clock",
        ":common",
        ":concurrent",
        ":os_util",
        ":shell",
        ":unix",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "profiler-output",
    srcs = glob([
        "com/google/devtools/build/lib/profiler/chart/*.java",
        "com/google/devtools/build/lib/profiler/output/*.java",
        "com/google/devtools/build/lib/profiler/statistics/*.java",
    ]),
    deps = [
        ":collect",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

# Clock classes.
java_library(
    name = "clock",
    srcs = [
        "com/google/devtools/build/lib/util/BlazeClock.java",
        "com/google/devtools/build/lib/util/Clock.java",
        "com/google/devtools/build/lib/util/JavaClock.java",
    ],
    deps = [
        ":concurrent",
    ],
)

# String canonicalizer.
java_library(
    name = "base-util",
    srcs = [
        "com/google/devtools/build/lib/util/StringCanonicalizer.java",
        "com/google/devtools/build/lib/util/StringTrie.java",
        "com/google/devtools/build/lib/util/VarInt.java",
    ],
    deps = [
        ":common",
        "//third_party:guava",
    ],
)

# I/O utilities.
java_library(
    name = "io",
    srcs = glob(["com/google/devtools/build/lib/util/io/*.java"]),
    deps = [
        ":clock",
        ":common",
        ":concurrent",
        ":vfs",
        "//third_party:guava",
    ],
)

# General utilities.
java_library(
    name = "os_util",
    srcs = ["com/google/devtools/build/lib/util/OS.java"],
)

java_library(
    name = "util",
    srcs = glob(
        ["com/google/devtools/build/lib/util/*.java"],
        exclude = [
            "com/google/devtools/build/lib/util/BlazeClock.java",
            "com/google/devtools/build/lib/util/Clock.java",
            "com/google/devtools/build/lib/util/JavaClock.java",
            "com/google/devtools/build/lib/util/OS.java",
            "com/google/devtools/build/lib/util/StringCanonicalizer.java",
            "com/google/devtools/build/lib/util/StringTrie.java",
            "com/google/devtools/build/lib/util/VarInt.java",
        ],
    ),
    deps = [
        ":base-util",
        ":clock",
        ":collect",
        ":common",
        ":concurrent",
        ":os_util",
        ":shell",
        ":unix",
        ":vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "skyframe-base",
    srcs = glob([
        "com/google/devtools/build/skyframe/*.java",
    ]),
    deps = [
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":util",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "analysis-exec-rules-skyframe",
    srcs = glob(
        [
            "com/google/devtools/build/lib/analysis/**/*.java",
            "com/google/devtools/build/lib/exec/*.java",
            "com/google/devtools/build/lib/ideinfo/**/*.java",
            "com/google/devtools/build/lib/rules/**/*.java",
            "com/google/devtools/build/lib/skyframe/*.java",
        ],
        exclude = [
            "com/google/devtools/build/lib/rules/java/JavaToolchainDataParser.java",
        ],
    ) + [
        "com/google/devtools/build/lib/runtime/BlazeServerStartupOptions.java",
    ],
    deps = [
        ":base-util",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":graph",
        ":io",
        ":os_util",
        ":packages",
        ":shell",
        ":skyframe-base",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:android_studio_ide_info_proto",
        "//src/main/protobuf:bundlemerge_proto",
        "//src/main/protobuf:crosstool_config_proto",
        "//src/main/protobuf:extra_actions_base_proto",
        "//src/main/protobuf:test_status_proto",
        "//src/main/protobuf:xcodegen_proto",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/common",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/util",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "graph",
    srcs = glob([
        "com/google/devtools/build/lib/graph/*.java",
    ]),
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "query2",
    srcs = glob([
        "com/google/devtools/build/lib/query2/**/*.java",
    ]),
    deps = [
        ":analysis-exec-rules-skyframe",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":graph",
        ":packages",
        ":skyframe-base",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "buildtool-runtime",
    srcs = glob([
        "com/google/devtools/build/lib/buildtool/**/*.java",
        "com/google/devtools/build/lib/runtime/**/*.java",
    ]),
    deps = [
        ":analysis-exec-rules-skyframe",
        ":clock",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":io",
        ":os_util",
        ":packages",
        ":profiler-output",
        ":query2",
        ":server",
        ":shell",
        ":skyframe-base",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/build/docgen",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//src/main/protobuf:invocation_policy_proto",
        "//src/main/protobuf:test_status_proto",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "runtime",
    srcs = glob([
        "com/google/devtools/build/lib/runtime/**/*.java",
        "com/google/devtools/build/lib/buildtool/**/*.java",
        "com/google/devtools/build/lib/server/**/*.java",
    ]),
    deps = [
        ":analysis-exec-rules-skyframe",
        ":clock",
        ":cmdline",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":io",
        ":os_util",
        ":packages",
        ":profiler-output",
        ":query2",
        ":shell",
        ":skyframe-base",
        ":unix",
        ":util",
        ":vfs",
        "//src/main/java/com/google/devtools/build/docgen",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//src/main/protobuf:invocation_policy_proto",
        "//src/main/protobuf:test_status_proto",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_library(
    name = "server",
    srcs = glob([
        "com/google/devtools/build/lib/server/**/*.java",
    ]),
    deps = [
        ":clock",
        ":io",
        ":unix",
        ":util",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "bazel-core",
    srcs = glob(
        [
            "com/google/devtools/build/lib/bazel/**/*.java",
        ],
        exclude = [
            "com/google/devtools/build/lib/bazel/repository/MavenConnector.java",
        ],
    ),
    resources = glob([
        "com/google/devtools/build/lib/**/*.txt",
        "com/google/devtools/build/lib/**/*.html",
        "com/google/devtools/build/lib/**/*.css",
        "com/google/devtools/build/lib/**/*.js",
        "com/google/devtools/build/lib/**/*.WORKSPACE",
        "com/google/devtools/build/lib/**/*.vm",
    ]) + [
        "//third_party:d3-js",
        "//third_party:jquery",
    ],
    visibility = [
        "//src:__subpackages__",
        "//src/tools/generate_workspace/src/main/java/com/google/devtools/build/workspace:__pkg__",
    ],
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
    ],
    deps = [
        ":analysis-exec-rules-skyframe",
        ":buildtool-runtime",
        ":clock",
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":graph",
        ":io",
        ":maven-connector",
        ":os_util",
        ":packages",
        ":query2",
        ":runtime",
        ":shell",
        ":skyframe-base",
        ":unix",
        ":util",
        ":vfs",
        "//src/java_tools/singlejar:zip",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/bazel/dash",
        "//src/main/java/com/google/devtools/build/lib/rules/genquery",
        "//src/main/java/com/google/devtools/build/lib/sandbox",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/build/lib/worker",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:build_proto",
        "//src/main/protobuf:crosstool_config_proto",
        "//src/main/protobuf:extra_actions_base_proto",
        "//src/main/protobuf:test_status_proto",
        "//src/main/protobuf:worker_protocol_proto",
        "//third_party:aether",
        "//third_party:apache_commons_compress",
        "//third_party:apache_commons_pool2",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jgit",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:maven",
        "//third_party:maven_model",
        "//third_party:plexus_component_annotations",
        "//third_party:protobuf",
        "//third_party:slf4j",
    ],
)

java_library(
    name = "maven-connector",
    srcs = ["com/google/devtools/build/lib/bazel/repository/MavenConnector.java"],
    deps = [
        "//third_party:aether",
        "//third_party:jsr330_inject",
        "//third_party:maven_model",
    ],
)

java_library(
    name = "java-toolchain-parser",
    testonly = 1,
    srcs = glob([
        # JavaToolchain parser
        "com/google/devtools/build/lib/rules/java/JavaToolchainData.java",
        "com/google/devtools/build/lib/rules/java/JavaToolchainDataParser.java",
        # Concurrent stuff
        "com/google/devtools/build/lib/concurrent/*.java",
    ]),
    visibility = [
        "//src/java_tools/buildjar:__pkg__",
    ],
    deps = [
        ":common",
        "//src/main/protobuf:build_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:protobuf",
    ],
)

java_binary(
    name = "bazel-main",
    main_class = "com.google.devtools.build.lib.bazel.BazelMain",
    visibility = ["//src:__pkg__"],
    runtime_deps = [
        ":bazel-core",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//src/main/java/com/google/devtools/common/options:srcs",
        "//src/main/java/com/google/devtools/build/lib/bazel/dash:srcs",
        "//src/main/java/com/google/devtools/build/lib/sandbox:srcs",
        "//src/main/java/com/google/devtools/build/lib/standalone:srcs",
        "//src/main/java/com/google/devtools/build/lib/worker:srcs",
        "//src/main/java/com/google/devtools/build/docgen:srcs",
    ],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "docs_embedded_in_sources",
    srcs = glob(["com/google/devtools/build/lib/**/*.java"]),
)

genrule(
    name = "gen_buildencyclopedia",
    srcs = [":docs_embedded_in_sources"],
    outs = ["build-encyclopedia.zip"],
    cmd = (
        "mkdir -p $(@D)/be && " +
        "$(location //src/main/java/com/google/devtools/build/docgen:docgen_bin)" +
        " $$PWD/src/main/java/com/google/devtools/build/lib" +
        " $(@D)/be &&" +
        "zip -qj $@ $(@D)/be/*"
    ),
    tools = [
        "//src/main/java/com/google/devtools/build/docgen:docgen_bin",
        "//src/main/java/com/google/devtools/build/docgen:docgen_javalib",
    ],
)

genrule(
    name = "gen_skylarklibrary",
    outs = ["skylark-library.zip"],
    cmd = "mkdir -p $(@D)/skylark-lib &&" +
          "$(location //src/main/java/com/google/devtools/build/docgen:skydoc_bin) $(@D)/skylark-lib &&" +
          "zip -qj $@ $(@D)/skylark-lib/*",
    tools = ["//src/main/java/com/google/devtools/build/docgen:skydoc_bin"],
)

#
# Bootstrapping SingleJar using Skylark rules
#
load("/tools/build_rules/java_rules_skylark", "bootstrap_java_library")

bootstrap_java_library(
    name = "shell-skylark",
    srcs = glob(["com/google/devtools/build/lib/shell/*.java"]),
    jars = ["//third_party:guava"],
)
