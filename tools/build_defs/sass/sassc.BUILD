package(default_visibility = ["//tools/build_defs/sass:__pkg__"])

BASE_DIR = "sassc-3.3.0-beta1/"

cc_binary(
    name = "sassc",
    srcs = [
        "@libsass//:srcs",
        BASE_DIR + "sassc.c",
    ],
    linkopts = ["-ldl -lm"],
    # TODO(perezd): Hack, is there a better way to reference this via libsass.BUILD?
    includes = ["../libsass/libsass-3.3.0-beta1/include"],
)
