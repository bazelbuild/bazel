def _wat_binary(ctx):
    out = ctx.actions.declare_file(ctx.attr.name + ".wasm")
    ctx.actions.run(
        executable = ctx.executable._wat2wasm,
        inputs = [ctx.file.src],
        outputs = [out],
        arguments = [
            ctx.file.src.path,
            "-o",
            out.path,
        ],
    )
    return DefaultInfo(files = depset([out]))

wat_binary = rule(
    implementation = _wat_binary,
    attrs = {
        "src": attr.label(
            allow_single_file = [".wat"],
            mandatory = True,
        ),
        "_wat2wasm": attr.label(
            default = "@wabt//src/tools:wat2wasm",
            executable = True,
            cfg = "exec",
        ),
    },
)
