java_library(
    name = "shell",
    srcs = glob(["com/google/devtools/build/lib/shell/*.java"]),
    visibility = ["//src:__subpackages__"],
    deps = ["//third_party:guava"],
)

java_library(
    name = "bazel-core",
    srcs = glob(
        ["**/*.java"],
        exclude = [
            "com/google/devtools/build/lib/shell/*.java",
            "com/google/devtools/build/lib/rules/java/JavaToolchainDataParser.java",
        ],
    ),
    resources = glob([
        "**/*.txt",
        "**/*.html",
        "**/*.css",
        "**/*.js",
    ]),
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
        ":shell",
        "//src/main/protobuf:proto_build",
        "//src/main/protobuf:proto_bundlemerge",
        "//src/main/protobuf:proto_crosstool_config",
        "//src/main/protobuf:proto_extra_actions_base",
        "//src/main/protobuf:proto_test_status",
        "//src/main/protobuf:proto_xcodegen",
        "//src/tools/xcode-common",
        "//third_party:aether",
        "//third_party:apache_commons_compress",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:joda-time",
        "//third_party:jsr305",
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
        "com/google/devtools/build/lib/Constants.java",
    ]),
    visibility = [
        "//src/test:__subpackages__",
    ],
    deps = [
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

# Build encyclopedia generation.
filegroup(
    name = "gen_be_sources",
    srcs = glob(["com/google/devtools/build/lib/**/*.java"]),
)

java_binary(
    name = "docgen_bin",
    srcs = glob(["com/google/devtools/build/docgen/*.java"]),
    main_class = "com.google.devtools.build.docgen.BuildEncyclopediaGenerator",
    resources = glob(
        ["com/google/devtools/build/docgen/templates/*.html"],
    ),
    deps = [
        ":bazel-core",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

genrule(
    name = "gen_buildencyclopedia",
    srcs = [":gen_be_sources"],
    outs = ["build-encyclopedia.html"],
    cmd = ("$(location docgen_bin)" +
           " $$PWD/src/main/java/com/google/devtools/build/lib" +
           " $$PWD" +
           "; cp $$PWD/build-encyclopedia.html $@"),
    tools = [":docgen_bin"],
)
