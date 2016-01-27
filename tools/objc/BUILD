package(default_visibility = ["//visibility:public"])

# Files which shouldn't be publicly visible and dependencies of all objc_* or ios_* rules shouldn't be in this package.
exports_files(glob(["**"]))

java_binary(
    name = "bundlemerge",
    main_class = "com.google.devtools.build.xcode.bundlemerge.BundleMerge",
    runtime_deps = [":bundlemerge_import"],
)

java_import(
    name = "bundlemerge_import",
    jars = [":precomp_bundlemerge_deploy.jar"],
)

java_binary(
    name = "plmerge",
    main_class = "com.google.devtools.build.xcode.plmerge.PlMerge",
    runtime_deps = [":plmerge_import"],
)

java_import(
    name = "plmerge_import",
    jars = [":precomp_plmerge_deploy.jar"],
)

java_binary(
    name = "xcodegen",
    main_class = "com.google.devtools.build.xcode.xcodegen.XcodeGen",
    runtime_deps = [":xcodegen_import"],
)

java_import(
    name = "xcodegen_import",
    jars = [":precomp_xcodegen_deploy.jar"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/objc/memleaks:srcs",
        "//tools/objc/sim_devices:BUILD",
    ],
)

filegroup(
    name = "testrunner",
    srcs = [":testrunner_stub"],
)

filegroup(
    name = "memleaks_plugin",
    srcs = [":memleaks_plugin_stub"],
)

filegroup(
    name = "dump_syms",
    srcs = [":dump_syms_stub"],
)

sh_binary(
    name = "ibtoolwrapper",
    srcs = [":ibtoolwrapper.sh"],
    data = [
        ":realpath",
        ":xcrunwrapper",
    ],
)

sh_binary(
    name = "actoolwrapper",
    srcs = [":actoolwrapper.sh"],
    data = [
        ":realpath",
        ":xcrunwrapper",
    ],
)

sh_binary(
    name = "momcwrapper",
    srcs = [":momcwrapper.sh"],
    data = [
        ":realpath",
        ":xcrunwrapper",
    ],
)

sh_binary(
    name = "swiftstdlibtoolwrapper",
    srcs = [":swiftstdlibtoolwrapper.sh"],
    data = [
        ":realpath",
        ":xcrunwrapper",
    ],
)

sh_binary(
    name = "xcrunwrapper",
    srcs = [":xcrunwrapper.sh"],
)

xcode_config(
    name = "host_xcodes",
)

# TODO(bazel-team): Open-source the script once J2ObjC support is open-sourced.
py_library(
    name = "j2objc_dead_code_pruner",
    srcs = ["j2objc_dead_code_pruner.py"],
)
