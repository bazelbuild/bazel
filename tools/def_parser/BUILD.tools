load("//tools:build_defs.bzl", "IS_HOST_WINDOWS")

package(default_visibility = ["//visibility:public"])

# This should use a toolchain setup like the one for //tools/launcher, but that
# doesn't provide a good way to bootstrap the def_parser from source build and
# targets such as `--malloc`, which must not depend on def_parser. Instead,
# the cc_* rules use a computed default and exclude tags. Since def_parser is
# only used when targeting Windows and itself only runs on Windows, we only
# match on the target platform here (which is the exec platform after the exec
# transition applied by the implicit dep edge from cc_* rules).
# TODO: Consider improving the situation by wrapping the targets with exclusions
#  with transitions.
filegroup(
    name = "def_parser",
    srcs = select({
        "@platforms//os:windows": ["def_parser.exe"] if IS_HOST_WINDOWS else ["//third_party/def_parser:def_parser"],
        "//conditions:default": ["no_op.bat"],
    }),
)
