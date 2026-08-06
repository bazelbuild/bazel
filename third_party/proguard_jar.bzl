"""Runs ProGuard and normalizes its output without shell tools."""

load("@rules_java//java:defs.bzl", "java_common")

def _proguard_jar_impl(ctx):
    java_runtime = ctx.attr._java_runtime[java_common.JavaRuntimeInfo]
    proguard_output = ctx.actions.declare_file(ctx.label.name + ".proguard.jar")

    java_args = ctx.actions.args()
    java_args.add("-Dlog4j.rootLogger=OFF")
    java_args.add("-jar")
    java_args.add(ctx.file.proguard)

    proguard_args = ctx.actions.args()
    proguard_args.add("-injars")
    proguard_args.add(ctx.file.src)
    proguard_args.add("-outjars")
    proguard_args.add(proguard_output)
    proguard_args.add("-libraryjars")
    proguard_args.add(ctx.file.library_jar)
    proguard_args.add(ctx.file.config, format = "@%s")

    ctx.actions.run(
        executable = java_runtime.java_executable_exec_path,
        arguments = [java_args, proguard_args],
        inputs = [
            ctx.file.config,
            ctx.file.library_jar,
            ctx.file.proguard,
            ctx.file.src,
        ],
        outputs = [proguard_output],
        tools = java_runtime.files,
        mnemonic = "Proguard",
        progress_message = "Stripping %{input} with ProGuard",
    )

    normalize_args = ctx.actions.args()
    normalize_args.add("d")
    normalize_args.add(proguard_output)
    normalize_args.add(ctx.outputs.out)
    normalize_args.add("__no_entries_match__")

    ctx.actions.run(
        executable = ctx.executable.zipper,
        arguments = [normalize_args],
        inputs = [proguard_output],
        outputs = [ctx.outputs.out],
        tools = [ctx.executable.zipper],
        mnemonic = "NormalizeJar",
        progress_message = "Normalizing %{output}",
    )

proguard_jar = rule(
    implementation = _proguard_jar_impl,
    attrs = {
        "src": attr.label(allow_single_file = True, mandatory = True),
        "library_jar": attr.label(allow_single_file = True, mandatory = True),
        "config": attr.label(allow_single_file = True, mandatory = True),
        "out": attr.output(mandatory = True),
        "proguard": attr.label(
            allow_single_file = True,
            cfg = "exec",
            mandatory = True,
        ),
        "zipper": attr.label(
            executable = True,
            cfg = "exec",
            mandatory = True,
        ),
        "_java_runtime": attr.label(
            cfg = "exec",
            default = "@bazel_tools//tools/jdk:current_java_runtime",
            providers = [java_common.JavaRuntimeInfo],
        ),
    },
)
