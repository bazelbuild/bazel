def _repo_rule_impl(ctx):
    ctx.file("BUILD", "")

my_repo = repository_rule(
    implementation = _repo_rule_impl,
    doc = "Minimal example of a repository rule.",
    attrs = {
        "useless": attr.string(
            doc = "This argument will be ignored. You don't have to specify it, but you may.",
            default = "ignoreme",
        ),
    },
)
