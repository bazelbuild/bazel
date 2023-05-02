# TODO: Remove on import. This file only exists to indicate the changes that have to be made to the
#       internal version of java_semantics.bzl.

def _get_java_runtime_dependent_runfiles_and_symlinks(
        ctx,
        *,
        executable,
        feature_config,
        is_absolute_path):
    java_runtime_toolchain = semantics.find_java_runtime_toolchain(ctx)

    # Add symlinks to the C++ runtime libraries under a path that can be built
    # into the Java binary without having to embed the crosstool, gcc, and grte
    # version information contained within the libraries' package paths.
    runfiles_symlinks = {}

    if not is_absolute_path(ctx, java_runtime_toolchain.java_home):
        runfiles_symlinks = {
            ("_cpp_runtimes/%s" % lib.basename): lib
            for lib in cc_helper.find_cpp_toolchain(ctx).dynamic_runtime_lib(
                feature_configuration = feature_config,
            ).to_list()
        }

    return [java_runtime_toolchain.files], runfiles_symlinks

semantics = struct(
    get_java_runtime_dependent_runfiles_and_symlinks = _get_java_runtime_dependent_runfiles_and_symlinks,
)
