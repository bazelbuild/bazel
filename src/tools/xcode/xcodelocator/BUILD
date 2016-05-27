package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
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
