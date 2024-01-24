# Copyright 2021 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Definition of java_import rule.
"""

load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/java/basic_java_library.bzl", "construct_defaultinfo")
load(":common/java/import_deps_check.bzl", "import_deps_check")
load(":common/java/java_common.bzl", "java_common")
load(":common/java/java_common_internal_for_builtins.bzl", _run_ijar_private_for_builtins = "run_ijar")
load(":common/java/java_info.bzl", "JavaInfo")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/java/proguard_validation.bzl", "validate_proguard_specs")

PackageSpecificationInfo = _builtins.toplevel.PackageSpecificationInfo

def _filter_provider(provider, *attrs):
    return [dep[provider] for attr in attrs for dep in attr if provider in dep]

def _collect_jars(ctx, jars):
    jars_dict = {}
    for info in jars:
        if JavaInfo in info:
            fail("'jars' attribute cannot contain labels of Java targets")
        for jar in info.files.to_list():
            jar_path = jar.dirname + jar.basename
            if jars_dict.get(jar_path) != None:
                fail("in jars attribute of java_import rule //" + ctx.label.package + ":" + ctx.attr.name + ": " + jar.basename + " is a duplicate")
            jars_dict[jar_path] = jar
    return [jar_tuple[1] for jar_tuple in jars_dict.items()] if len(jars_dict.items()) > 0 else []

def _process_with_ijars_if_needed(jars, ctx):
    file_dict = {}
    use_ijars = ctx.fragments.java.use_ijars()
    for jar in jars:
        interface_jar = jar
        if use_ijars:
            ijar_basename = jar.short_path.removeprefix("../").removesuffix("." + jar.extension) + "-ijar.jar"
            interface_jar_directory = "_ijar/" + ctx.label.name + "/" + ijar_basename

            interface_jar = ctx.actions.declare_file(interface_jar_directory)
            _run_ijar_private_for_builtins(
                ctx.actions,
                target_label = ctx.label,
                jar = jar,
                output = interface_jar,
                java_toolchain = semantics.find_java_toolchain(ctx),
            )
        file_dict[jar] = interface_jar

    return file_dict

def _check_export_error(ctx, exports):
    not_in_allowlist = hasattr(ctx.attr, "_allowlist_java_import_exports") and not getattr(ctx.attr, "_allowlist_java_import_exports")[PackageSpecificationInfo].contains(ctx.label)
    disallow_java_import_exports = ctx.fragments.java.disallow_java_import_exports()

    if len(exports) != 0 and (disallow_java_import_exports or not_in_allowlist):
        fail("java_import.exports is no longer supported; use java_import.deps instead")

def _check_empty_jars_error(ctx, jars):
    # TODO(kotlaja): Remove temporary incompatible flag [disallow_java_import_empty_jars] once migration is done.
    not_in_allowlist = hasattr(ctx.attr, "_allowlist_java_import_empty_jars") and not getattr(ctx.attr, "_allowlist_java_import_empty_jars")[PackageSpecificationInfo].contains(ctx.label)
    disallow_java_import_empty_jars = ctx.fragments.java.disallow_java_import_empty_jars()

    if len(jars) == 0 and disallow_java_import_empty_jars and not_in_allowlist:
        fail("empty java_import.jars is no longer supported " + ctx.label.package)

def _create_java_info_with_dummy_output_file(ctx, srcjar, all_deps, exports, runtime_deps_list, neverlink, cc_info_list, add_exports, add_opens):
    dummy_jar = ctx.actions.declare_file(ctx.label.name + "_dummy.jar")
    dummy_src_jar = srcjar
    if dummy_src_jar == None:
        dummy_src_jar = ctx.actions.declare_file(ctx.label.name + "_src_dummy.java")
        ctx.actions.write(dummy_src_jar, "")
    return java_common.compile(
        ctx,
        output = dummy_jar,
        java_toolchain = semantics.find_java_toolchain(ctx),
        source_files = [dummy_src_jar],
        deps = all_deps,
        runtime_deps = runtime_deps_list,
        neverlink = neverlink,
        exports = [export[JavaInfo] for export in exports if JavaInfo in export],  # Watchout, maybe you need to add them there manually.
        native_libraries = cc_info_list,
        add_exports = add_exports,
        add_opens = add_opens,
    )

def bazel_java_import_rule(
        ctx,
        jars = [],
        srcjar = None,
        deps = [],
        runtime_deps = [],
        exports = [],
        neverlink = False,
        proguard_specs = [],
        add_exports = [],
        add_opens = []):
    """Implements java_import.

    This rule allows the use of precompiled .jar files as libraries in other Java rules.

    Args:
      ctx: (RuleContext) Used to register the actions.
      jars: (list[Artifact]) List of output jars.
      srcjar: (Artifact) The jar containing the sources.
      deps: (list[Target]) The list of dependent libraries.
      runtime_deps: (list[Target]) Runtime dependencies to attach to the rule.
      exports: (list[Target])  The list of exported libraries.
      neverlink: (bool) Whether this rule should only be used for compilation and not at runtime.
      constraints: (list[String]) Rule constraints.
      proguard_specs: (list[File]) Files to be used as Proguard specification.
      add_exports: (list[str]) Allow this library to access the given <module>/<package>.
      add_opens: (list[str]) Allow this library to reflectively access the given <module>/<package>.

    Returns:
      (list[provider]) A list containing DefaultInfo, JavaInfo,
      OutputGroupsInfo, ProguardSpecProvider providers.
    """

    _check_empty_jars_error(ctx, jars)
    _check_export_error(ctx, exports)

    collected_jars = _collect_jars(ctx, jars)
    all_deps = _filter_provider(JavaInfo, deps, exports)

    jdeps_artifact = None
    merged_java_info = java_common.merge(all_deps)
    not_in_allowlist = hasattr(ctx.attr, "_allowlist_java_import_deps_checking") and not ctx.attr._allowlist_java_import_deps_checking[PackageSpecificationInfo].contains(ctx.label)
    if len(collected_jars) > 0 and not_in_allowlist and "incomplete-deps" not in ctx.attr.tags:
        jdeps_artifact = import_deps_check(
            ctx,
            collected_jars,
            merged_java_info.compile_jars,
            merged_java_info.transitive_compile_time_jars,
            "java_import",
        )

    compilation_to_runtime_jar_map = _process_with_ijars_if_needed(collected_jars, ctx)
    runtime_deps_list = [runtime_dep[JavaInfo] for runtime_dep in runtime_deps if JavaInfo in runtime_dep]
    cc_info_list = [dep[CcInfo] for dep in deps if CcInfo in dep]
    java_info = None
    if len(collected_jars) > 0:
        java_infos = []
        for jar in collected_jars:
            java_infos.append(JavaInfo(
                output_jar = jar,
                compile_jar = compilation_to_runtime_jar_map[jar],
                deps = all_deps,
                runtime_deps = runtime_deps_list,
                neverlink = neverlink,
                source_jar = srcjar,
                exports = [export[JavaInfo] for export in exports if JavaInfo in export],  # Watchout, maybe you need to add them there manually.
                native_libraries = cc_info_list,
                add_exports = add_exports,
                add_opens = add_opens,
            ))
        java_info = java_common.merge(java_infos)
    else:
        # TODO(kotlaja): Remove next line once all java_import targets with empty jars attribute are cleaned from depot (b/246559727).
        java_info = _create_java_info_with_dummy_output_file(ctx, srcjar, all_deps, exports, runtime_deps_list, neverlink, cc_info_list, add_exports, add_opens)

    target = {"JavaInfo": java_info}

    target["ProguardSpecProvider"] = validate_proguard_specs(
        ctx,
        proguard_specs,
        [deps, runtime_deps, exports],
    )

    # TODO(kotlaja): Revise if collected_runtimes can be added into construct_defaultinfo directly.
    collected_runtimes = []
    for runtime_dep in ctx.attr.runtime_deps:
        collected_runtimes.extend(runtime_dep.files.to_list())

    target["DefaultInfo"] = construct_defaultinfo(
        ctx,
        collected_jars,
        collected_jars + collected_runtimes,
        neverlink,
        exports,
    )

    output_group_src_jars = depset() if srcjar == None else depset([srcjar])
    target["OutputGroupInfo"] = OutputGroupInfo(
        **{
            "_source_jars": output_group_src_jars,
            "_direct_source_jars": output_group_src_jars,
            "_validation": depset() if jdeps_artifact == None else depset([jdeps_artifact]),
            "_hidden_top_level_INTERNAL_": target["ProguardSpecProvider"].specs,
        }
    )
    return target

def _proxy(ctx):
    return bazel_java_import_rule(
        ctx,
        ctx.attr.jars,
        ctx.file.srcjar,
        ctx.attr.deps,
        ctx.attr.runtime_deps,
        ctx.attr.exports,
        ctx.attr.neverlink,
        ctx.files.proguard_specs,
        ctx.attr.add_exports,
        ctx.attr.add_opens,
    ).values()

_ALLOWED_RULES_IN_DEPS_FOR_JAVA_IMPORT = [
    "java_library",
    "java_import",
    "cc_library",
    "cc_binary",
]

JAVA_IMPORT_ATTRS = {
    "data": attr.label_list(
        allow_files = True,
        flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
        doc = """
The list of files needed by this rule at runtime.
        """,
    ),
    "deps": attr.label_list(
        providers = [JavaInfo],
        allow_rules = _ALLOWED_RULES_IN_DEPS_FOR_JAVA_IMPORT,
        doc = """
The list of other libraries to be linked in to the target.
See <a href="${link java_library.deps}">java_library.deps</a>.
        """,
    ),
    "exports": attr.label_list(
        providers = [JavaInfo],
        allow_rules = _ALLOWED_RULES_IN_DEPS_FOR_JAVA_IMPORT,
        doc = """
Targets to make available to users of this rule.
See <a href="${link java_library.exports}">java_library.exports</a>.
        """,
    ),
    "runtime_deps": attr.label_list(
        allow_files = [".jar"],
        allow_rules = _ALLOWED_RULES_IN_DEPS_FOR_JAVA_IMPORT,
        providers = [[CcInfo], [JavaInfo]],
        flags = ["SKIP_ANALYSIS_TIME_FILETYPE_CHECK"],
        doc = """
Libraries to make available to the final binary or test at runtime only.
See <a href="${link java_library.runtime_deps}">java_library.runtime_deps</a>.
        """,
    ),
    # JavaImportBazeRule attr
    "jars": attr.label_list(
        allow_files = [".jar"],
        mandatory = True,
        doc = """
The list of JAR files provided to Java targets that depend on this target.
        """,
    ),
    "srcjar": attr.label(
        allow_single_file = [".srcjar", ".jar"],
        flags = ["DIRECT_COMPILE_TIME_INPUT"],
        doc = """
A JAR file that contains source code for the compiled JAR files.
        """,
    ),
    "neverlink": attr.bool(
        default = False,
        doc = """
Only use this library for compilation and not at runtime.
Useful if the library will be provided by the runtime environment
during execution. Examples of libraries like this are IDE APIs
for IDE plug-ins or <code>tools.jar</code> for anything running on
a standard JDK.
        """,
    ),
    "constraints": attr.string_list(
        doc = """
Extra constraints imposed on this rule as a Java library.
        """,
    ),
    # ProguardLibraryRule attr
    "proguard_specs": attr.label_list(
        allow_files = True,
        doc = """
Files to be used as Proguard specification.
These will describe the set of specifications to be used by Proguard. If specified,
they will be added to any <code>android_binary</code> target depending on this library.

The files included here must only have idempotent rules, namely -dontnote, -dontwarn,
assumenosideeffects, and rules that start with -keep. Other options can only appear in
<code>android_binary</code>'s proguard_specs, to ensure non-tautological merges.
        """,
    ),
    # Additional attrs
    "add_exports": attr.string_list(
        doc = """
Allow this library to access the given <code>module</code> or <code>package</code>.
<p>
This corresponds to the javac and JVM --add-exports= flags.
        """,
    ),
    "add_opens": attr.string_list(
        doc = """
Allow this library to reflectively access the given <code>module</code> or
<code>package</code>.
<p>
This corresponds to the javac and JVM --add-opens= flags.
        """,
    ),
    "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
    "_java_toolchain_type": attr.label(default = semantics.JAVA_TOOLCHAIN_TYPE),
}

java_import = rule(
    _proxy,
    doc = """
<p>
  This rule allows the use of precompiled <code>.jar</code> files as
  libraries for <code><a href="${link java_library}">java_library</a></code> and
  <code>java_binary</code> rules.
</p>

<h4 id="java_import_examples">Examples</h4>

<pre class="code">
<code class="lang-starlark">
    java_import(
        name = "maven_model",
        jars = [
            "maven_model/maven-aether-provider-3.2.3.jar",
            "maven_model/maven-model-3.2.3.jar",
            "maven_model/maven-model-builder-3.2.3.jar",
        ],
    )
</code>
</pre>
    """,
    attrs = JAVA_IMPORT_ATTRS,
    provides = [JavaInfo],
    fragments = ["java", "cpp"],
    toolchains = [semantics.JAVA_TOOLCHAIN],
)
