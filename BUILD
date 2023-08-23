# Bazel - Google's Build System

load("//tools/distributions:distribution_rules.bzl", "distrib_jar_filegroup")
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_license//rules:license.bzl", "license")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

package(default_visibility = ["//scripts/release:__pkg__"])

license(
    name = "license",
    package_name = "bazelbuild/bazel",
    copyright_notice = "Copyright © 2014 The Bazel Authors. All rights reserved.",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
)

exports_files(["LICENSE"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["*"],
        exclude = [
            "WORKSPACE",  # Needs to be filtered.
            "bazel-*",  # convenience symlinks
            "out",  # IntelliJ with setup-intellij.sh
            "output",  # output of compile.sh
            ".*",  # mainly .git* files
        ],
    ) + [
        "//:WORKSPACE.filtered",
        "//examples:srcs",
        "//scripts:srcs",
        "//site:srcs",
        "//src:srcs",
        "//src/main/java/com/google/devtools/build/docgen/release:srcs",
        "//src/main/starlark/tests/builtins_bzl:srcs",
        "//third_party:srcs",
        "//tools:srcs",
    ] + glob([".bazelci/*"]) + [
        ".bazelrc",
        ".bazelversion",
    ],
    applicable_licenses = ["@io_bazel//:license"],
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "git",
    srcs = glob(
        [".git/**"],
        allow_empty = True,
        exclude = [".git/**/*[*"],  # gitk creates temp files with []
    ),
)

filegroup(
    name = "dummy",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "workspace-file",
    srcs = [
        ":WORKSPACE",
        ":distdir.bzl",
        ":distdir_deps.bzl",
    ],
    visibility = [
        "//src/test/shell/bazel:__subpackages__",
    ],
)

filegroup(
    name = "changelog-file",
    srcs = [":CHANGELOG.md"],
    visibility = [
        "//scripts/packages:__subpackages__",
    ],
)

genrule(
    name = "filtered_WORKSPACE",
    srcs = ["WORKSPACE"],
    outs = ["WORKSPACE.filtered"],
    cmd = "\n".join([
        "cp $< $@",
        # Comment out the android repos if they exist.
        "sed -i.bak -e 's/^android_sdk_repository/# android_sdk_repository/' -e 's/^android_ndk_repository/# android_ndk_repository/' $@",
    ]),
)

pkg_tar(
    name = "bootstrap-jars",
    srcs = [
        "@blake3",
        "@com_google_protobuf//:protobuf_java",
        "@com_google_protobuf//:protobuf_java_util",
        "@com_google_protobuf//:protobuf_javalite",
        "@zstd-jni//:zstd-jni",
    ],
    package_dir = "derived/jars",
    strip_prefix = "external",
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)

distrib_jar_filegroup(
    name = "bootstrap-derived-java-jars",
    srcs = glob(
        ["derived/jars/**/*.jar"],
        allow_empty = True,
    ),
    enable_distributions = ["debian"],
    visibility = ["//:__subpackages__"],
)

filegroup(
    name = "bootstrap-derived-java-srcs",
    srcs = glob(
        ["derived/**/*.java"],
        allow_empty = True,
    ),
    visibility = ["//:__subpackages__"],
)

# Additional generated files that are not Java sources (which could otherwise
# be included in //src:derived_java_sources).
filegroup(
    name = "generated_resources",
    srcs = [
        "//src/main/java/com/google/devtools/build/lib/bazel/rules:builtins_bzl.zip",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules:coverage.WORKSPACE",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules:rules_license.WORKSPACE",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/cpp:cc_configure.WORKSPACE",
        "//src/main/java/com/google/devtools/build/lib/bazel/rules/java:jdk.WORKSPACE",
    ],
)

pkg_tar(
    name = "bazel-srcs",
    srcs = [
        ":generated_resources",
        ":srcs",
    ],
    # TODO(aiuto): Replace with pkg_filegroup when that is available.
    remap_paths = {
        "WORKSPACE.filtered": "WORKSPACE",
        # Rewrite paths coming from local repositories back into third_party.
        "external/googleapis": "third_party/googleapis",
        "external/remoteapis": "third_party/remoteapis",
    },
    strip_prefix = ".",
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)

pkg_tar(
    name = "platforms-srcs",
    srcs = ["@platforms//:srcs"],
    strip_prefix = "external",
    visibility = ["//:__subpackages__"],
)

pkg_tar(
    name = "rules_java-srcs",
    srcs = ["@rules_java//:distribution"],
    strip_prefix = "external",
    visibility = ["//:__subpackages__"],
)

# The @maven repository is created by maven_install from rules_jvm_external.
# `@maven//:srcs` contains all jar files downloaded and BUILD files created by maven_install.
pkg_tar(
    name = "maven-srcs",
    srcs = ["@maven//:srcs"],
    strip_prefix = "external",
    visibility = ["//:__subpackages__"],
)

exports_files(
    ["maven_install.json"],
    visibility = ["//tools/compliance:__pkg__"],
)

py_binary(
    name = "combine_distfiles",
    srcs = ["combine_distfiles.py"],
    visibility = ["//visibility:private"],
    deps = ["//src:create_embedded_tools_lib"],
)

genrule(
    name = "bazel-distfile",
    srcs = [
        ":bazel-srcs",
        ":bootstrap-jars",
        ":platforms-srcs",
        ":rules_java-srcs",
        ":maven-srcs",
        "//src:derived_java_srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:bootstrap_autocodec.tar",
        "@additional_distfiles//:archives.tar",
    ],
    outs = ["bazel-distfile.zip"],
    cmd = "$(location :combine_distfiles) $@ $(SRCS)",
    tools = [":combine_distfiles"],
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)

genrule(
    name = "bazel-distfile-tar",
    srcs = [
        ":bazel-srcs",
        ":bootstrap-jars",
        ":platforms-srcs",
        ":rules_java-srcs",
        ":maven-srcs",
        "//src:derived_java_srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:bootstrap_autocodec.tar",
        "@additional_distfiles//:archives.tar",
    ],
    outs = ["bazel-distfile.tar"],
    cmd = "$(location :combine_distfiles_to_tar.sh) $@ $(SRCS)",
    tools = ["combine_distfiles_to_tar.sh"],
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)

constraint_setting(name = "machine_size")

# A machine with "high cpu count".
constraint_value(
    name = "highcpu_machine",
    constraint_setting = ":machine_size",
    visibility = ["//visibility:public"],
)

platform(
    name = "default_host_platform",
    constraint_values = [
        ":highcpu_machine",
    ],
    parents = ["@local_config_platform//:host"],
)

platform(
    name = "windows_arm64",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:arm64",
    ],
)

REMOTE_PLATFORMS = ("rbe_ubuntu1804_java11",)

[
    platform(
        name = platform_name + "_platform",
        exec_properties = {
            "dockerNetwork": "standard",
            "dockerPrivileged": "true",
            "Pool": "default",
        },
        parents = ["@" + platform_name + "//config:platform"],
    )
    for platform_name in REMOTE_PLATFORMS
]

[
    # The highcpu RBE platform where heavy actions run on. In order to
    # use this platform add the highcpu_machine constraint to your target.
    platform(
        name = platform_name + "_highcpu_platform",
        constraint_values = [
            "//:highcpu_machine",
        ],
        exec_properties = {
            "Pool": "highcpu",
        },
        parents = ["//:" + platform_name + "_platform"],
    )
    for platform_name in REMOTE_PLATFORMS
]
