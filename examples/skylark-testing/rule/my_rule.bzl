def _impl(ctx):
    s = set([file for dep in ctx.attr.deps for file in dep.files])
    return struct(my_rule_provider = s)

my_rule = rule(_impl,
    attrs = { 'deps' : attr.label_list() }
)