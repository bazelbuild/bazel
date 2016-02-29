package(default_visibility = ["//src:__subpackages__"])

cc_library(
    name = "network-tools",
    srcs = ["network-tools.c"],
    hdrs = ["network-tools.h"],
    copts = ["-std=c99"],
    deps = [":process-tools"],
)

cc_library(
    name = "process-tools",
    srcs = ["process-tools.c"],
    hdrs = ["process-tools.h"],
    copts = ["-std=c99"],
)

cc_binary(
    name = "process-wrapper",
    srcs = ["process-wrapper.c"],
    copts = ["-std=c99"],
    linkopts = ["-lm"],
    deps = [":process-tools"],
)

cc_binary(
    name = "build-runfiles",
    srcs = ["build-runfiles.cc"],
)

cc_binary(
    name = "namespace-sandbox",
    srcs = select({
        "//src:darwin": ["namespace-sandbox-dummy.c"],
        "//src:darwin_x86_64": ["namespace-sandbox-dummy.c"],
        "//src:freebsd": ["namespace-sandbox-dummy.c"],
        "//src:windows": ["namespace-sandbox-dummy.c"],
        "//conditions:default": ["namespace-sandbox.c"],
    }),
    copts = ["-std=c99"],
    linkopts = ["-lm"],
    deps = select({
        "//src:darwin": [],
        "//src:darwin_x86_64": [],
        "//src:freebsd": [],
        "//src:windows": [],
        "//conditions:default": [
            ":process-tools",
            ":network-tools",
        ],
    }),
)

DARWIN_XCODE_LOCATOR_COMPILE_COMMAND = """
  /usr/bin/xcrun clang -fobjc-arc -framework CoreServices \
      -framework Foundation -o $@ $<
"""

genrule(
    name = "xcode-locator-genrule",
    srcs = select({
        "//src:darwin": ["xcode_locator.m"],
        "//src:darwin_x86_64": ["xcode_locator.m"],
        "//conditions:default": ["xcode_locator_stub.sh"],
    }),
    outs = ["xcode-locator"],
    cmd = select({
        "//src:darwin": DARWIN_XCODE_LOCATOR_COMPILE_COMMAND,
        "//src:darwin_x86_64": DARWIN_XCODE_LOCATOR_COMPILE_COMMAND,
        "//conditions:default": "cp $< $@",
    }),
    local = 1,
    output_to_bindir = 1,
)

filegroup(
    name = "jdk-support",
    srcs = [
        "jdk.BUILD",
    ],
)

exports_files([
    "build_interface_so",
])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)
