package(default_visibility = ["//visibility:public"])

load("//tools/build_defs/sass:sass.bzl", "sass_binary")

# Import our shared colors and fonts so we can generate a CSS file.
sass_binary(
    name = "hello_world",
    src = "main.scss",
    deps = [
        "//examples/sass/shared:colors",
        "//examples/sass/shared:fonts",
    ],
)
