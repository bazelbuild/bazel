# Bazel - Google's Build System

package(default_visibility = ["//scripts/release:__pkg__"])

exports_files(["LICENSE"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["*"],
        exclude = [
            "bazel-*",  # convenience symlinks
            "out",  # IntelliJ with setup-intellij.sh
            "output",  # output of compile.sh
            ".*",  # mainly .git* files
        ],
    ) + [
        "//examples:srcs",
        "//scripts:srcs",
        "//site:srcs",
        "//src:srcs",
        "//tools:srcs",
        "//third_party:srcs",
    ] + glob([".bazelci/*"]),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)

filegroup(
    name = "git",
    srcs = glob(
        [".git/**"],
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

filegroup(
    name = "bootstrap-derived-java-srcs",
    srcs = glob(["derived/**/*.java"]),
    visibility = ["//:__subpackages__"],
)

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "bazel-srcs",
    srcs = [":srcs"],
    remap_paths = {
        # Rewrite paths coming from local repositories back into third_party.
        "../googleapis": "third_party/googleapis",
        "../remoteapis": "third_party/remoteapis",
    },
    strip_prefix = ".",
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
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

# This is a workaround for fetching Bazel toolchains, for remote execution.
# See https://github.com/bazelbuild/bazel/issues/3246.
# Will be removed once toolchain fetching is supported.
filegroup(
    name = "dummy_toolchain_reference",
    srcs = ["@bazel_toolchains//configs/debian8_clang/0.2.0/bazel_0.9.0:empty"],
    visibility = ["//visibility:public"],
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
    parents = ["@bazel_tools//platforms:host_platform"],
    contraint_values = [
        ":highcpu_machine",
    ],
)

# The default RBE platform where most actions run on
platform(
    name = "rbe_ubuntu1604_default",
    parents = ["@bazel_toolchains//configs/ubuntu16_04_clang/1.2:rbe_ubuntu1604"],
    remote_execution_properties = """
        properties: {
          name: "container-image"
          value:"docker://gcr.io/bazel-untrusted/ubuntu1604_bazel_rbe@sha256:b60b641427ca8aa99134cc1f9889e3349e391eed85854b5cfbb462884ec3420b"
        }
        properties: {
          name: "dockerNetwork"
          value: "standard"
        }
        properties: {
          name: "dockerPrivileged"
          value: "true"
        }
        """,
)

# The highcpu RBE platform where heavy actions run on. In order to
# use this platform add the highcpu_machine constraint to your target.
platform(
    name = "rbe_ubuntu1604_highcpu",
    parents = [":rbe_ubuntu1604_default"],
    constraint_values = [
        ":highcpu_machine",
    ],
    remote_execution_properties = """
        {PARENT_REMOTE_EXECUTION_PROPERTIES}
        properties: {
          name: "gceMachineType"
          value: "n1-highcpu-32"
        }
        """,
)
