package(default_visibility = ["//tools/build_defs/sass:__pkg__"])

BASE_DIR = "sassc-3.3.0-beta1/"

cc_binary(
    name = "sassc",
    srcs = [
        "@libsass//:srcs",
        BASE_DIR + "sassc.c",
    ],
    linkopts = ["-ldl", "-lm"],
    deps = ["@libsass//:headers"],
)
