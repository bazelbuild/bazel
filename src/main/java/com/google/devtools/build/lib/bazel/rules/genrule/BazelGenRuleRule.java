// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.bazel.rules.genrule;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.rules.genrule.GenRuleBaseRule;

/**
 * Rule definition for genrule for Bazel.
 */
public final class BazelGenRuleRule implements RuleDefinition {
  public static final String GENRULE_SETUP_LABEL = "//tools/genrule:genrule-setup.sh";

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    /* <!-- #BLAZE_RULE(genrule).NAME -->
    <br/>You may refer to this rule by name in the
    <code>srcs</code> or <code>deps</code> section of other <code>BUILD</code>
    rules. If the rule generates source files, you should use the
    <code>srcs</code> attribute.
    <!-- #END_BLAZE_RULE.NAME --> */
    return builder
        .setOutputToGenfiles()
        .add(
            attr("$genrule_setup", LABEL)
                .cfg(HostTransition.createFactory())
                .value(env.getToolsLabel(GENRULE_SETUP_LABEL)))

        // TODO(bazel-team): stamping doesn't seem to work. Fix it or remove attribute.
        .add(attr("stamp", BOOLEAN).value(false))
        .build();
  }

  @Override
  public RuleDefinition.Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("genrule")
        .ancestors(GenRuleBaseRule.class)
        .factoryClass(BazelGenRule.class)
        .build();
  }

}

/*<!-- #BLAZE_RULE (NAME = genrule, FAMILY = General)[GENERIC_RULE] -->

<p>A <code>genrule</code> generates one or more files using a user-defined Bash command.</p>

<p>
  Genrules are generic build rules that you can use if there's no specific rule for the task. If for
  example you want to minify JavaScript files then you can use a genrule to do so. If however you
  need to compile C++ files, stick to the existing <code>cc_*</code> rules, because all the heavy
  lifting has already been done for you.
</p>
<p>
  Do not use a genrule for running tests. There are special dispensations for tests and test
  results, including caching policies and environment variables. Tests generally need to be run
  after the build is complete and on the target architecture, whereas genrules are executed during
  the build and on the host architecture (the two may be different). If you need a general purpose
  testing rule, use <a href="${link sh_test}"><code>sh_test</code></a>.
</p>

<h4>Cross-compilation Considerations</h4>

<p>
  <em>See <a href="../user-manual.html#configurations">the user manual</a> for more info about
  cross-compilation.</em>
</p>
<p>
  While genrules run during a build, their outputs are often used after the build, for deployment or
  testing. Consider the example of compiling C code for a microcontroller: the compiler accepts C
  source files and generates code that runs on a microcontroller. The generated code obviously
  cannot run on the CPU that was used for building it, but the C compiler (if compiled from source)
  itself has to.
</p>
<p>
  The build system uses the host configuration to describe the machine(s) on which the build runs
  and the target configuration to describe the machine(s) on which the output of the build is
  supposed to run. It provides options to configure each of these and it segregates the
  corresponding files into separate directories to avoid conflicts.
</p>
<p>
  For genrules, the build system ensures that dependencies are built appropriately:
  <code>srcs</code> are built (if necessary) for the <em>target</em> configuration,
  <code>tools</code> are built for the <em>host</em> configuration, and the output is considered to
  be for the <em>target</em> configuration. It also provides <a href="${link make-variables}">
  "Make" variables</a> that genrule commands can pass to the corresponding tools.
</p>
<p>
  It is intentional that genrule defines no <code>deps</code> attribute: other built-in rules use
  language-dependent meta information passed between the rules to automatically determine how to
  handle dependent rules, but this level of automation is not possible for genrules. Genrules work
  purely at the file and runfiles level.
</p>

<h4>Special Cases</h4>

<p>
  <i>Host-host compilation</i>: in some cases, the build system needs to run genrules such that the
  output can also be executed during the build. If for example a genrule builds some custom compiler
  which is subsequently used by another genrule, the first one has to produce its output for the
  host configuration, because that's where the compiler will run in the other genrule. In this case,
  the build system does the right thing automatically: it builds the <code>srcs</code> and
  <code>outs</code> of the first genrule for the host configuration instead of the target
  configuration. See <a href="../user-manual.html#configurations">the user manual</a> for more
  info.
</p>
<p>
  <i>JDK & C++ Tooling</i>: to use a tool from the JDK or the C++ compiler suite, the build system
  provides a set of variables to use. See <a href="${link make-variables}">"Make" variable</a> for
  details.
</p>

<h4>Genrule Environment</h4>

<p>
  The genrule command is executed by a Bash shell that is configured to fail when a command
  or a pipeline fails, using <code>set -e -o pipefail</code>.
</p>
<p>
  The build tool executes the Bash command in a sanitized process environment that
  defines only core variables such as <code>PATH</code>, <code>PWD</code>,
  <code>TMPDIR</code>, and a few others.

  To ensure that builds are reproducible, most variables defined in the user's shell
  environment are not passed though to the genrule's command. However, Bazel (but not
  Blaze) passes through the value of the user's <code>PATH</code> environment variable.

  Any change to the value of <code>PATH</code> will cause Bazel to re-execute the command
  on the next build.
  <!-- See https://github.com/bazelbuild/bazel/issues/1142 -->
</p>
<p>
  A genrule command should not access the network except to connect processes that are
  children of the command itself, though this is not currently enforced.
</p>
<p>
  The build system automatically deletes any existing output files, but creates any necessary parent
  directories before it runs a genrule. It also removes any output files in case of a failure.
</p>

<h4>General Advice</h4>

<ul>
  <li>Do ensure that tools run by a genrule are deterministic and hermetic. They should not write
    timestamps to their output, and they should use stable ordering for sets and maps, as well as
    write only relative file paths to the output, no absolute paths. Not following this rule will
    lead to unexpected build behavior (Bazel not rebuilding a genrule you thought it would) and
    degrade cache performance.</li>
  <li>Do use <code>$(location)</code> extensively, for outputs, tools and sources. Due to the
    segregation of output files for different configurations, genrules cannot rely on hard-coded
    and/or absolute paths.</li>
  <li>Do write a common Starlark macro in case the same or very similar genrules are used in
    multiple places. If the genrule is complex, consider implementing it in a script or as a
    Starlark rule. This improves readability as well as testability.</li>
  <li>Do make sure that the exit code correctly indicates success or failure of the genrule.</li>
  <li>Do not write informational messages to stdout or stderr. While useful for debugging, this can
    easily become noise; a successful genrule should be silent. On the other hand, a failing genrule
    should emit good error messages.</li>
  <li><code>$$</code> evaluates to a <code>$</code>, a literal dollar-sign, so in order to invoke a
    shell command containing dollar-signs such as <code>ls $(dirname $x)</code>, one must escape it
    thus: <code>ls $$(dirname $$x)</code>.</li>
  <li>Avoid creating symlinks and directories. Bazel doesn't copy over the directory/symlink
    structure created by genrules and its dependency checking of directories is unsound.</li>
  <li>When referencing the genrule in other rules, you can use either the genrule's label or the
    labels of individual output files. Sometimes the one approach is more readable, sometimes the
    other: referencing outputs by name in a consuming rule's <code>srcs</code> will avoid
    unintentionally picking up other outputs of the genrule, but can be tedious if the genrule
    produces many outputs.</li>
</ul>

<h4 id="genrule_examples">Examples</h4>

<p>
  This example generates <code>foo.h</code>. There are no sources, because the command doesn't take
  any input. The "binary" run by the command is a perl script in the same package as the genrule.
</p>
<pre class="code">
genrule(
    name = "foo",
    srcs = [],
    outs = ["foo.h"],
    cmd = "./$(location create_foo.pl) &gt; \"$@\"",
    tools = ["create_foo.pl"],
)
</pre>

<p>
  The following example shows how to use a <a href="${link filegroup}"><code>filegroup</code>
  </a> and the outputs of another <code>genrule</code>. Note that using <code>$(SRCS)</code> instead
  of explicit <code>$(location)</code> directives would also work; this example uses the latter for
  sake of demonstration.
</p>
<pre class="code">
genrule(
    name = "concat_all_files",
    srcs = [
        "//some:files",  # a filegroup with multiple files in it ==> $(location<b>s</b>)
        "//other:gen",   # a genrule with a single output ==> $(location)
    ],
    outs = ["concatenated.txt"],
    cmd = "cat $(locations //some:files) $(location //other:gen) > $@",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
