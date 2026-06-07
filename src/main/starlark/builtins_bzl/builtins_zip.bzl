"""Hermetic builtins archive rule."""

def _builtins_zip_impl(ctx):
    args = ctx.actions.args()
    args.add("cC")
    args.add(ctx.outputs.out)

    package_marker = ctx.label.package + "/"
    for src in sorted(ctx.files.srcs, key = lambda file: file.short_path):
        marker_index = src.short_path.find(package_marker)
        if marker_index < 0:
            fail("builtins source is outside {}: {}".format(ctx.label.package, src.short_path))
        relative_path = src.short_path[marker_index + len(package_marker):]
        args.add("builtins_bzl/{}={}".format(relative_path, src.path))

    ctx.actions.run(
        executable = ctx.executable.zipper,
        arguments = [args],
        inputs = ctx.files.srcs,
        outputs = [ctx.outputs.out],
        mnemonic = "BuiltinsZip",
        progress_message = "Building %{output}",
        tools = [ctx.executable.zipper],
    )

builtins_zip = rule(
    implementation = _builtins_zip_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "out": attr.output(mandatory = True),
        "zipper": attr.label(
            executable = True,
            cfg = "exec",
            mandatory = True,
        ),
    },
)
