def _test_rule(ctx):
    out = ctx.actions.declare_file("out.txt")
    files = ctx.attr.module_source.files
    found = False
    for file_ in files:
        if file_.basename == "package.json":
            compare_version(ctx.actions, file_, out, ctx.attr.version)
            found = True
            break
    if not found:
        fail("Not found package.json")
    return [DefaultInfo(files = depset([out]))]

test_rule = rule(
    implementation = _test_rule,
    attrs = {
        "module_source": attr.label(),
        "version": attr.string(),
    },
)

def compare_version(action_factory, file_, out, expected_version):
    action_factory.run_shell(
        mnemonic = "getVersion",
        inputs = [file_],
        outputs = [out],
        command = """result=$(cat ./{file} | grep '"version": "{expected}"' || exit 1) \
&& echo $result > ./{out}""".format(
            file = file_.path,
            out = out.path,
            expected = expected_version,
        ),
    )
