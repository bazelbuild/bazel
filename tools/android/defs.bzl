def _run_ijar(ctx):
    ijar_jar = java_common.run_ijar(
        ctx.actions,
        jar = ctx.file.jar,
        java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],
    )
    return [DefaultInfo(files = depset([ijar_jar]))]

run_ijar = rule(
    implementation = _run_ijar,
    attrs = {
        "jar": attr.label(mandatory = True, allow_single_file = True),
        "_java_toolchain": attr.label(
            default = "//tools/jdk:current_java_toolchain",
            providers = [java_common.JavaRuntimeInfo],
        ),
    },
)
