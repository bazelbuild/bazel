package(default_visibility = ["//tools/build_defs/sass:__pkg__"])

cc_binary(
    name = "sassc",
    srcs = [
        "@libsass//:srcs",
        "sassc.c",
        "sassc_version.h",
],
    linkopts = ["-ldl", "-lm"],
    deps = ["@libsass//:headers"],
)
