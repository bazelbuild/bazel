load("@rules_java//java/common:java_info.bzl", "JavaInfo")

def _extract_source_jars_impl(ctx):
    source_jars = []
    for dep in ctx.attr.deps:
        if JavaInfo in dep:
            source_jars.extend(dep[JavaInfo].source_jars)

    if not source_jars:
        fail("No source jars found in dependencies")

    return [DefaultInfo(files = depset(source_jars))]

extract_source_jars = rule(
    implementation = _extract_source_jars_impl,
    attrs = {
        "deps": attr.label_list(providers = [JavaInfo]),
    },
)
