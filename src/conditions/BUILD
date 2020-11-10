filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = ["BUILD.tools"],
    visibility = ["//src:__pkg__"],
)

config_setting(
    name = "linux",
    constraint_values = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_arm",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:arm",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:ppc",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc64le",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:ppc",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_s390x",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:s390x",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    constraint_values = ["@platforms//os:macos"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_arm64e",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64e",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "iphonesdk",
    values = {"define": "IPHONE_SDK=1"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "freebsd",
    constraint_values = ["@platforms//os:freebsd"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "openbsd",
    constraint_values = ["@platforms//os:openbsd"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm",
    constraint_values = ["@platforms//cpu:arm"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "remote",
    values = {"define": "EXECUTOR=remote"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debian_build",
    values = {
        "define": "distribution=debian",
    },
    visibility = ["//visibility:public"],
)
