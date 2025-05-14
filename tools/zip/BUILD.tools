load("@platforms//host:constraints.bzl", "HOST_CONSTRAINTS")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "zipper",
    srcs = select({
        ":matches_host_constraints": glob(["zipper/*"]),
        "//conditions:default": ["//third_party/ijar:zipper"],
    }),
)

alias(
    name = "unzip_fdo",
    actual = "zipper",
)

config_setting(
    name = "matches_host_constraints",
    constraint_values = HOST_CONSTRAINTS,
    visibility = ["//visibility:private"],
)
