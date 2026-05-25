"""Apply proguard rules to a JAR file."""

def _proguard_jar_impl(ctx):
    inputs = ctx.files.srcs + ctx.files.deps + [ctx.file.proguard_spec]
    output = ctx.outputs.out

    args = ctx.actions.args()
    args.add_joined("--srcs", ctx.files.srcs, join_with = ",")
    args.add_joined("--deps", ctx.files.deps, join_with = ",")
    args.add("--proguard_spec", ctx.file.proguard_spec)
    args.add("--output", output)
    args.add("--timestamp", "1980-01-01 00:00:00")

    ctx.actions.run(
        inputs = inputs,
        mnemonic = "ProguardJar",
        outputs = [output],
        executable = ctx.executable._wrapper,
        arguments = [args],
    )

    return DefaultInfo(files = depset([output]))

proguard_jar = rule(
    implementation = _proguard_jar_impl,
    attrs = {
        "srcs": attr.label_list(),
        "deps": attr.label_list(),
        "proguard_spec": attr.label(allow_single_file = True),
        "out": attr.output(),
        "_wrapper": attr.label(
            cfg = "exec",
            default = ":wrapper_private",
            executable = True,
        ),
    },
)
