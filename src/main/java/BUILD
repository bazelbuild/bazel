package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "shell",
    srcs = glob(["com/google/devtools/build/lib/shell/*.java"]),
    deps = ["//third_party:guava"],
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
    name = "options",
    srcs = glob(["com/google/devtools/common/options/*.java"]),
    visibility = ["//visibility:public"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "vfs",
    srcs = glob([
        "com/google/devtools/build/lib/vfs/**/*.java",
        "com/google/devtools/build/lib/profiler/**/*.java",
        "com/google/devtools/build/lib/util/**/*.java",
    ]),
    deps = [
        ":collect",
        ":common",
        ":concurrent",
        ":options",
        ":shell",
        ":unix",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "events",
    srcs = glob(["com/google/devtools/build/lib/events/*.java"]),
    deps = [
        ":common",
        ":concurrent",
        ":vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "bazel-core",
    srcs = glob(
        [
            "com/google/devtools/build/lib/exec/**",
            "com/google/devtools/build/lib/server/**",
            "com/google/devtools/build/lib/server/**",
            "com/google/devtools/build/lib/pkgcache/**",
            "com/google/devtools/build/lib/query2/**",
            "com/google/devtools/build/lib/bazel/**/*.java",
            "com/google/devtools/build/lib/actions/**",
            "com/google/devtools/build/lib/graph/**",
            "com/google/devtools/build/lib/cmdline/**",
            "com/google/devtools/build/lib/skyframe/**",
            "com/google/devtools/build/lib/webstatusserver/**/*.java",
            "com/google/devtools/build/lib/analysis/**",
            "com/google/devtools/build/lib/rules/**",
            "com/google/devtools/build/lib/syntax/**",
            "com/google/devtools/build/docgen/**/*.java",
            "com/google/devtools/build/lib/packages/**",
            "com/google/devtools/build/lib/runtime/**/*.java",
            "com/google/devtools/build/lib/standalone/**",
            "com/google/devtools/build/lib/buildtool/**",
            "com/google/devtools/build/skyframe/**",
        ],
        exclude = [
            "com/google/devtools/build/lib/rules/java/JavaToolchainDataParser.java",
        ],
    ),
    resources = glob([
        "com/google/devtools/build/lib/**/*.txt",
        "com/google/devtools/build/lib/**/*.html",
        "com/google/devtools/build/lib/**/*.css",
        "com/google/devtools/build/lib/**/*.js",
    ]) + [
        "//third_party:d3-js",
        "//third_party:jquery",
    ],
    visibility = ["//src/test/java:__subpackages__"],
    runtime_deps = [
        "//third_party:aether",
        "//third_party:apache_commons_logging",
        "//third_party:apache_httpclient",
        "//third_party:apache_httpcore",
        "//third_party:maven_model",
        "//third_party:plexus_interpolation",
        "//third_party:plexus_utils",
    ],
    deps = [
        ":collect",
        ":common",
        ":concurrent",
        ":events",
        ":options",
        ":shell",
        ":unix",
        ":vfs",
        "//src/main/protobuf:proto_build",
        "//src/main/protobuf:proto_bundlemerge",
        "//src/main/protobuf:proto_crosstool_config",
        "//src/main/protobuf:proto_extra_actions_base",
        "//src/main/protobuf:proto_test_status",
        "//src/main/protobuf:proto_xcodegen",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/common",
        "//src/tools/xcode-common/java/com/google/devtools/build/xcode/util",
        "//third_party:aether",
        "//third_party:apache_commons_compress",
        "//third_party:auto_value",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:joda_time",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
        "//third_party:maven_model",
        "//third_party:protobuf",
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
        "//src/main/protobuf:proto_build",
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
    srcs = glob(["**"]),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "be-template_files",
    srcs = glob(["com/google/devtools/build/docgen/templates/**/*.html"]),
)

java_library(
    name = "docgen_javalib",
    srcs = glob(
        ["com/google/devtools/build/docgen/*.java"],
        exclude = [
            "com/google/devtools/build/docgen/BuildEncyclopediaGenerator.java",
            "com/google/devtools/build/docgen/SkylarkDocumentationGenerator.java",
        ],
    ),
    resources = [":be-template_files"],
    exports = [
        ":bazel-core",
    ],
    deps = [
        ":bazel-core",
        ":common",
        "//src/main/java:vfs",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "docs_embedded_in_sources",
    srcs = glob(["com/google/devtools/build/lib/**/*.java"]),
)

java_binary(
    name = "docgen_bin",
    srcs = ["com/google/devtools/build/docgen/BuildEncyclopediaGenerator.java"],
    main_class = "com.google.devtools.build.docgen.BuildEncyclopediaGenerator",
    deps = [
        ":common",
        ":docgen_javalib",
    ],
)

java_binary(
    name = "skydoc_bin",
    srcs = ["com/google/devtools/build/docgen/SkylarkDocumentationGenerator.java"],
    main_class = "com.google.devtools.build.docgen.SkylarkDocumentationGenerator",
    deps = [":docgen_javalib"],
)

genrule(
    name = "gen_buildencyclopedia",
    srcs = [":docs_embedded_in_sources"],
    outs = ["build-encyclopedia.html"],
    cmd = ("$(location docgen_bin)" +
           " $$PWD/src/main/java/com/google/devtools/build/lib" +
           " $$PWD" +
           " && cp $$PWD/build-encyclopedia.html $@"),
    tools = [":docgen_bin"],
)

genrule(
    name = "gen_skylarklibrary",
    outs = ["skylark-library.html"],
    cmd = "$(location :skydoc_bin) $@",
    tools = [":skydoc_bin"],
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
