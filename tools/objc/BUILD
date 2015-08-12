package(default_visibility = ["//visibility:public"])

# Files which shouldn't be publicly visible and dependencies of all objc_* or ios_* rules shouldn't be in this package.
exports_files(glob(["**"]))

java_binary(
    name = "actoolzip",
    srcs = [":precomp_actoolzip_deploy.jar"],
    main_class = "com.google.devtools.build.xcode.actoolzip.ActoolZip",
)

java_binary(
    name = "ibtoolzip",
    srcs = [":precomp_ibtoolzip_deploy.jar"],
    main_class = "com.google.devtools.build.xcode.ibtoolzip.IbtoolZip",
)

java_binary(
    name = "bundlemerge",
    srcs = [":precomp_bundlemerge_deploy.jar"],
    main_class = "com.google.devtools.build.xcode.bundlemerge.BundleMerge",
)

java_binary(
    name = "momczip",
    srcs = [":precomp_momczip_deploy.jar"],
    main_class = "com.google.devtools.build.xcode.momczip.MomcZip",
)

java_binary(
    name = "plmerge",
    srcs = [":precomp_plmerge_deploy.jar"],
    main_class = "com.google.devtools.build.xcode.plmerge.PlMerge",
)

java_binary(
    name = "xcodegen",
    srcs = [":precomp_xcodegen_deploy.jar"],
    main_class = "com.google.devtools.build.xcode.xcodegen.XcodeGen",
)

java_binary(
    name = "swiftstdlibtoolzip",
    srcs = [":precomp_swiftstdlibtoolzip_deploy.jar"],
    main_class = "com.google.devtools.build.xcode.swiftstdlibtoolzip.SwiftStdlibToolZip",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "testrunner",
)

sh_binary(
    name = "ibtoolwrapper",
    srcs = [":ibtoolwrapper.sh"],
    data = [":realpath"],
)

sh_binary(
    name = "actoolwrapper",
    srcs = [":actoolwrapper.sh"],
    data = [":realpath"],
)

sh_binary(
    name = "momcwrapper",
    srcs = [":momcwrapper.sh"],
    data = [":realpath"],
)

sh_binary(
    name = "swiftstdlibtoolwrapper",
    srcs = [":swiftstdlibtoolwrapper.sh"],
    data = [":realpath"],
)

# TODO(bazel-team): Open-source the script once J2ObjC support is open-sourced.
py_library(
    name = "j2objc_dead_code_pruner",
    srcs = ["j2objc_dead_code_pruner.py"],
)
