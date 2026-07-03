"""Hermetic builtins archive rule."""

def _format_zip_entry(src):
    package_prefix = src.owner.package + "/"
    relative_path = src.short_path.partition(package_prefix)[2]
    return "builtins_bzl/{}={}".format(relative_path, src.path)

def _builtins_zip_impl(ctx):
    args = ctx.actions.args()
    args.add("cC")
    args.add(ctx.outputs.out)
    args.add_all(ctx.files.srcs, map_each = _format_zip_entry)

    ctx.actions.run(
        executable = ctx.executable._zipper,
        arguments = [args],
        inputs = ctx.files.srcs,
        outputs = [ctx.outputs.out],
        mnemonic = "BuiltinsZip",
        progress_message = "Building %{output}",
    )

builtins_zip = rule(
    implementation = _builtins_zip_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "out": attr.output(mandatory = True),
        "_zipper": attr.label(
            default = "//third_party/ijar:zipper",
            executable = True,
            cfg = "exec",
        ),
    },
)
