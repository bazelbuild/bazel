# Bazel - Google's Build System

load("@bazel_skylib//rules:write_file.bzl", "write_file")
load("@rules_license//rules:license.bzl", "license")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_binary")
load("//src/tools/bzlmod:utils.bzl", "get_canonical_repo_name")
load("//tools/distributions:distribution_rules.bzl", "distrib_jar_filegroup")

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
            "MODULE.bazel.lock",  # Use MODULE.bazel.lock.dist instead
            "WORKSPACE.bzlmod",  # Needs to be filtered.
            "bazel-*",  # convenience symlinks
            "out",  # IntelliJ with setup-intellij.sh
            "output",  # output of compile.sh
            ".*",  # mainly .git* files
        ],
    ) + [
        "//:MODULE.bazel.lock.dist",
        "//:WORKSPACE.bzlmod.filtered",
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
    name = "dummy",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "workspace-file",
    srcs = [
        ":WORKSPACE",
        ":distdir.bzl",
        ":workspace_deps.bzl",
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
    srcs = ["WORKSPACE.bzlmod"],
    outs = ["WORKSPACE.bzlmod.filtered"],
    cmd = "\n".join([
        "cp $< $@",
        # Comment out the android repos if they exist.
        "sed -i.bak -e 's/^android_sdk_repository/# android_sdk_repository/' -e 's/^android_ndk_repository/# android_ndk_repository/' $@",
    ]),
)

genrule(
    name = "generate_dist_lockfile",
    srcs = [
        "MODULE.bazel",
        "//third_party/googleapis:MODULE.bazel",
        "//third_party/remoteapis:MODULE.bazel",
    ],
    outs = ["MODULE.bazel.lock.dist"],
    cmd = " && ".join([
        "ROOT=$$PWD",
        "TMPDIR=$$(mktemp -d)",
        "trap 'rm -rf $$TMPDIR' EXIT",
        "mkdir -p $$TMPDIR/workspace",
        "touch $$TMPDIR/workspace/BUILD.bazel",
        "for i in $(SRCS); do dir=$$TMPDIR/workspace/$$(dirname $$i); mkdir -p $$dir; cp $$i $$dir; done",
        "cd $$TMPDIR/workspace",
        # Instead of `bazel mod deps`, we run a simpler command like `bazel query :all` here
        # so that we only trigger module resolution, not extension eval.
        # Also use `--batch` so that Bazel doesn't keep a server process alive.
        "$$ROOT/$(location //src:bazel) --batch --output_user_root=$$TMPDIR/output_user_root query --check_direct_dependencies=error --lockfile_mode=update :all",
        "mv MODULE.bazel.lock $$ROOT/$@",
    ]),
    tags = ["requires-network"],
    tools = ["//src:bazel"],
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
        "MODULE.bazel.lock.dist": "MODULE.bazel.lock",
        "WORKSPACE.bzlmod.filtered": "WORKSPACE.bzlmod",
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

write_file(
    name = "gen_maven_repo_name",
    out = "MAVEN_CANONICAL_REPO_NAME",
    # TODO: Use this instead after building with Bazel 7.1.0 or later.
    #    content = [get_canonical_repo_name("@maven")],
    content = ["rules_jvm_external~~maven~maven"],
)

# The @maven repository is created by maven_install from rules_jvm_external.
# `@maven//:srcs` contains all jar files downloaded and BUILD files created by maven_install.
pkg_tar(
    name = "maven-srcs",
    srcs = ["@maven//:srcs"] + ["MAVEN_CANONICAL_REPO_NAME"],
    package_dir = "derived/maven",
    strip_prefix = "external/" + get_canonical_repo_name("@maven"),
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
        ":maven-srcs",
        "//src:derived_java_srcs",
        "//src/main/java/com/google/devtools/build/lib/skyframe/serialization/autocodec:bootstrap_autocodec.tar",
        "@bootstrap_repo_cache//:archives.tar",
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
        "@bootstrap_repo_cache//:archives.tar",
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

REMOTE_PLATFORMS = ("rbe_ubuntu2004",)

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
