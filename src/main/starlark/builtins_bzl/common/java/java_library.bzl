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
Definition of java_library rule.
"""

load(":common/cc/cc_info.bzl", "CcInfo")
load(":common/java/android_lint.bzl", "android_lint_subrule")
load(":common/java/basic_java_library.bzl", "BASIC_JAVA_LIBRARY_IMPLICIT_ATTRS", "basic_java_library", "construct_defaultinfo")
load(":common/java/boot_class_path_info.bzl", "BootClassPathInfo")
load(":common/java/java_info.bzl", "JavaInfo", "JavaPluginInfo")
load(":common/java/java_semantics.bzl", "semantics")
load(":common/rule_util.bzl", "merge_attrs")

def bazel_java_library_rule(
        ctx,
        srcs = [],
        deps = [],
        runtime_deps = [],
        plugins = [],
        exports = [],
        exported_plugins = [],
        resources = [],
        javacopts = [],
        neverlink = False,
        proguard_specs = [],
        add_exports = [],
        add_opens = [],
        bootclasspath = None,
        javabuilder_jvm_flags = None):
    """Implements java_library.

    Use this call when you need to produce a fully fledged java_library from
    another rule's implementation.

    Args:
      ctx: (RuleContext) Used to register the actions.
      srcs: (list[File]) The list of source files that are processed to create the target.
      deps: (list[Target]) The list of other libraries to be linked in to the target.
      runtime_deps: (list[Target]) Libraries to make available to the final binary or test at runtime only.
      plugins: (list[Target]) Java compiler plugins to run at compile-time.
      exports: (list[Target]) Exported libraries.
      exported_plugins: (list[Target]) The list of `java_plugin`s (e.g. annotation
        processors) to export to libraries that directly depend on this library.
      resources: (list[File]) A list of data files to include in a Java jar.
      javacopts: (list[str]) Extra compiler options for this library.
      neverlink: (bool) Whether this library should only be used for compilation and not at runtime.
      proguard_specs: (list[File]) Files to be used as Proguard specification.
      add_exports: (list[str]) Allow this library to access the given <module>/<package>.
      add_opens: (list[str]) Allow this library to reflectively access the given <module>/<package>.
      bootclasspath: (Target) The JDK APIs to compile this library against.
      javabuilder_jvm_flags: (list[str]) Additional JVM flags to pass to JavaBuilder.
    Returns:
      (dict[str, provider]) A list containing DefaultInfo, JavaInfo,
        InstrumentedFilesInfo, OutputGroupsInfo, ProguardSpecProvider providers.
    """
    if not srcs and deps:
        fail("deps not allowed without srcs; move to runtime_deps?")

    target, base_info = basic_java_library(
        ctx,
        srcs,
        deps,
        runtime_deps,
        plugins,
        exports,
        exported_plugins,
        resources,
        [],  # resource_jars
        [],  # class_pathresources
        javacopts,
        neverlink,
        proguard_specs = proguard_specs,
        add_exports = add_exports,
        add_opens = add_opens,
        bootclasspath = bootclasspath,
        javabuilder_jvm_flags = javabuilder_jvm_flags,
    )

    target["DefaultInfo"] = construct_defaultinfo(
        ctx,
        base_info.files_to_build,
        base_info.runfiles,
        neverlink,
        exports,
        runtime_deps,
    )
    target["OutputGroupInfo"] = OutputGroupInfo(**base_info.output_groups)

    return target

def _proxy(ctx):
    return bazel_java_library_rule(
        ctx,
        ctx.files.srcs,
        ctx.attr.deps,
        ctx.attr.runtime_deps,
        ctx.attr.plugins,
        ctx.attr.exports,
        ctx.attr.exported_plugins,
        ctx.files.resources,
        ctx.attr.javacopts,
        ctx.attr.neverlink,
        ctx.files.proguard_specs,
        ctx.attr.add_exports,
        ctx.attr.add_opens,
        ctx.attr.bootclasspath,
        ctx.attr.javabuilder_jvm_flags,
    ).values()

JAVA_LIBRARY_IMPLICIT_ATTRS = BASIC_JAVA_LIBRARY_IMPLICIT_ATTRS

JAVA_LIBRARY_ATTRS = merge_attrs(
    JAVA_LIBRARY_IMPLICIT_ATTRS,
    {
        "srcs": attr.label_list(
            allow_files = [".java", ".srcjar", ".properties"] + semantics.EXTRA_SRCS_TYPES,
            flags = ["DIRECT_COMPILE_TIME_INPUT", "ORDER_INDEPENDENT"],
            doc = """
The list of source files that are processed to create the target.
This attribute is almost always required; see exceptions below.
<p>
Source files of type <code>.java</code> are compiled. In case of generated
<code>.java</code> files it is generally advisable to put the generating rule's name
here instead of the name of the file itself. This not only improves readability but
makes the rule more resilient to future changes: if the generating rule generates
different files in the future, you only need to fix one place: the <code>outs</code> of
the generating rule. You should not list the generating rule in <code>deps</code>
because it is a no-op.
</p>
<p>
Source files of type <code>.srcjar</code> are unpacked and compiled. (This is useful if
you need to generate a set of <code>.java</code> files with a genrule.)
</p>
<p>
Rules: if the rule (typically <code>genrule</code> or <code>filegroup</code>) generates
any of the files listed above, they will be used the same way as described for source
files.
</p>
<p>
Source files of type <code>.properties</code> are treated as resources.
</p>

<p>All other files are ignored, as long as there is at least one file of a
file type described above. Otherwise an error is raised.</p>

<p>
This argument is almost always required, except if you specify the <code>runtime_deps</code> argument.
</p>
            """,
        ),
        "data": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
            doc = """
The list of files needed by this library at runtime.
See general comments about <code>data</code> at
<a href="${link common-definitions#typical-attributes}">Typical attributes defined by
most build rules</a>.
<p>
  When building a <code>java_library</code>, Bazel doesn't put these files anywhere; if the
  <code>data</code> files are generated files then Bazel generates them. When building a
  test that depends on this <code>java_library</code> Bazel copies or links the
  <code>data</code> files into the runfiles area.
</p>
            """ + semantics.DOCS.for_attribute("data"),
        ),
        "resources": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE", "ORDER_INDEPENDENT"],
            doc = """
A list of data files to include in a Java jar.
<p>
Resources may be source files or generated files.
</p>
            """ + semantics.DOCS.for_attribute("resources"),
        ),
        "plugins": attr.label_list(
            providers = [JavaPluginInfo],
            allow_files = True,
            cfg = "exec",
            doc = """
Java compiler plugins to run at compile-time.
Every <code>java_plugin</code> specified in this attribute will be run whenever this rule
is built. A library may also inherit plugins from dependencies that use
<code><a href="#java_library.exported_plugins">exported_plugins</a></code>. Resources
generated by the plugin will be included in the resulting jar of this rule.
            """,
        ),
        "deps": attr.label_list(
            allow_files = [".jar"],
            allow_rules = semantics.ALLOWED_RULES_IN_DEPS + semantics.ALLOWED_RULES_IN_DEPS_WITH_WARNING,
            providers = [
                [CcInfo],
                [JavaInfo],
            ],
            flags = ["SKIP_ANALYSIS_TIME_FILETYPE_CHECK"],
            doc = """
The list of libraries to link into this library.
See general comments about <code>deps</code> at
<a href="${link common-definitions#typical-attributes}">Typical attributes defined by
most build rules</a>.
<p>
  The jars built by <code>java_library</code> rules listed in <code>deps</code> will be on
  the compile-time classpath of this rule. Furthermore the transitive closure of their
  <code>deps</code>, <code>runtime_deps</code> and <code>exports</code> will be on the
  runtime classpath.
</p>
<p>
  By contrast, targets in the <code>data</code> attribute are included in the runfiles but
  on neither the compile-time nor runtime classpath.
</p>
            """,
        ),
        "runtime_deps": attr.label_list(
            allow_files = [".jar"],
            allow_rules = semantics.ALLOWED_RULES_IN_DEPS,
            providers = [[CcInfo], [JavaInfo]],
            flags = ["SKIP_ANALYSIS_TIME_FILETYPE_CHECK"],
            doc = """
Libraries to make available to the final binary or test at runtime only.
Like ordinary <code>deps</code>, these will appear on the runtime classpath, but unlike
them, not on the compile-time classpath. Dependencies needed only at runtime should be
listed here. Dependency-analysis tools should ignore targets that appear in both
<code>runtime_deps</code> and <code>deps</code>.
            """,
        ),
        "exports": attr.label_list(
            allow_rules = semantics.ALLOWED_RULES_IN_DEPS,
            providers = [[JavaInfo], [CcInfo]],
            doc = """
Exported libraries.
<p>
  Listing rules here will make them available to parent rules, as if the parents explicitly
  depended on these rules. This is not true for regular (non-exported) <code>deps</code>.
</p>
<p>
  Summary: a rule <i>X</i> can access the code in <i>Y</i> if there exists a dependency
  path between them that begins with a <code>deps</code> edge followed by zero or more
  <code>exports</code> edges. Let's see some examples to illustrate this.
</p>
<p>
  Assume <i>A</i> depends on <i>B</i> and <i>B</i> depends on <i>C</i>. In this case
  C is a <em>transitive</em> dependency of A, so changing C's sources and rebuilding A will
  correctly rebuild everything. However A will not be able to use classes in C. To allow
  that, either A has to declare C in its <code>deps</code>, or B can make it easier for A
  (and anything that may depend on A) by declaring C in its (B's) <code>exports</code>
  attribute.
</p>
<p>
  The closure of exported libraries is available to all direct parent rules. Take a slightly
  different example: A depends on B, B depends on C and D, and also exports C but not D.
  Now A has access to C but not to D. Now, if C and D exported some libraries, C' and D'
  respectively, A could only access C' but not D'.
</p>
<p>
  Important: an exported rule is not a regular dependency. Sticking to the previous example,
  if B exports C and wants to also use C, it has to also list it in its own
  <code>deps</code>.
</p>
            """,
        ),
        "exported_plugins": attr.label_list(
            providers = [JavaPluginInfo],
            cfg = "exec",
            doc = """
The list of <code><a href="#${link java_plugin}">java_plugin</a></code>s (e.g. annotation
processors) to export to libraries that directly depend on this library.
<p>
  The specified list of <code>java_plugin</code>s will be applied to any library which
  directly depends on this library, just as if that library had explicitly declared these
  labels in <code><a href="${link java_library.plugins}">plugins</a></code>.
</p>
            """,
        ),
        "bootclasspath": attr.label(
            providers = [BootClassPathInfo],
            flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
            doc = """Restricted API, do not use!""",
        ),
        "javabuilder_jvm_flags": attr.string_list(doc = """Restricted API, do not use!"""),
        "javacopts": attr.string_list(
            doc = """
Extra compiler options for this library.
Subject to <a href="make-variables.html">"Make variable"</a> substitution and
<a href="common-definitions.html#sh-tokenization">Bourne shell tokenization</a>.
<p>These compiler options are passed to javac after the global compiler options.</p>
            """,
        ),
        "neverlink": attr.bool(
            doc = """
Whether this library should only be used for compilation and not at runtime.
Useful if the library will be provided by the runtime environment during execution. Examples
of such libraries are the IDE APIs for IDE plug-ins or <code>tools.jar</code> for anything
running on a standard JDK.
<p>
  Note that <code>neverlink = 1</code> does not prevent the compiler from inlining material
  from this library into compilation targets that depend on it, as permitted by the Java
  Language Specification (e.g., <code>static final</code> constants of <code>String</code>
  or of primitive types). The preferred use case is therefore when the runtime library is
  identical to the compilation library.
</p>
<p>
  If the runtime library differs from the compilation library then you must ensure that it
  differs only in places that the JLS forbids compilers to inline (and that must hold for
  all future versions of the JLS).
</p>
            """,
        ),
        "resource_strip_prefix": attr.string(
            doc = """
The path prefix to strip from Java resources.
<p>
If specified, this path prefix is stripped from every file in the <code>resources</code>
attribute. It is an error for a resource file not to be under this directory. If not
specified (the default), the path of resource file is determined according to the same
logic as the Java package of source files. For example, a source file at
<code>stuff/java/foo/bar/a.txt</code> will be located at <code>foo/bar/a.txt</code>.
</p>
            """,
        ),
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
    },
)

java_library = rule(
    _proxy,
    doc = """
<p>This rule compiles and links sources into a <code>.jar</code> file.</p>

<h4>Implicit outputs</h4>
<ul>
  <li><code>lib<var>name</var>.jar</code>: A Java archive containing the class files.</li>
  <li><code>lib<var>name</var>-src.jar</code>: An archive containing the sources ("source
    jar").</li>
</ul>
    """,
    attrs = JAVA_LIBRARY_ATTRS,
    provides = [JavaInfo],
    outputs = {
        "classjar": "lib%{name}.jar",
        "sourcejar": "lib%{name}-src.jar",
    },
    fragments = ["java", "cpp"],
    toolchains = [semantics.JAVA_TOOLCHAIN],
    subrules = [android_lint_subrule],
)
