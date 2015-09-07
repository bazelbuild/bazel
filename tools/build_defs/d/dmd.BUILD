package(default_visibility = ["//visibility:public"])

config_setting(
    name = "darwin",
    values = {"host_cpu": "darwin"},
)

config_setting(
    name = "k8",
    values = {"host_cpu": "k8"},
)

filegroup(
    name = "dmd",
    srcs = select({
        ":darwin": ["dmd2/osx/bin/dmd"],
        ":k8": ["dmd2/linux/bin64/dmd"],
    }),
)

filegroup(
    name = "libphobos2",
    srcs = select({
        ":darwin": ["dmd2/osx/lib/libphobos2.a"],
        ":k8": [
            "dmd2/linux/lib64/libphobos2.a",
            "dmd2/linux/lib64/libphobos2.so",
        ],
    }),
)

filegroup(
    name = "phobos-src",
    srcs = glob(["dmd2/src/phobos/**/*.*"]),
)

filegroup(
    name = "druntime-import-src",
    srcs = glob([
        "dmd2/src/druntime/import/*.*",
        "dmd2/src/druntime/import/**/*.*",
    ]),
)
