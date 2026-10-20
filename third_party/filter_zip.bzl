"""Hermetic ZIP filtering rule."""

def _filter_zip_impl(ctx):
    args = ctx.actions.args()
    args.add("d")
    args.add(ctx.file.src)
    args.add(ctx.outputs.out)
    args.add_all(ctx.attr.exclude_patterns)

    ctx.actions.run(
        executable = ctx.executable.zipper,
        arguments = [args],
        inputs = [ctx.file.src],
        outputs = [ctx.outputs.out],
        mnemonic = "FilterZip",
        progress_message = "Filtering %{input}",
        tools = [ctx.executable.zipper],
    )

filter_zip = rule(
    implementation = _filter_zip_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "exclude_patterns": attr.string_list(),
        "out": attr.output(mandatory = True),
        "zipper": attr.label(
            executable = True,
            cfg = "exec",
            mandatory = True,
        ),
    },
)
