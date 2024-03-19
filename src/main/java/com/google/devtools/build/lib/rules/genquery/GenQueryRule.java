// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.genquery;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.GENQUERY_SCOPE_TYPE_LIST;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Types.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;

/** Rule definition for genquery the rule. */
public final class GenQueryRule implements RuleDefinition {

  /** Adds {@link GenQueryRule} and its dependencies to the provided builder. */
  public static void register(ConfiguredRuleClassProvider.Builder builder) {
    builder.addConfigurationFragment(GenQueryConfiguration.class);
    builder.addRuleDefinition(new GenQueryRule());
  }

  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment env) {
    return builder
        .requiresConfigurationFragments(GenQueryConfiguration.class)
        /* <!-- #BLAZE_RULE(genquery).ATTRIBUTE(scope) -->
        The scope of the query. The query is not allowed to touch targets outside the transitive
        closure of these targets.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("scope", GENQUERY_SCOPE_TYPE_LIST).mandatory().legacyAllowAnyFileType())
        /* <!-- #BLAZE_RULE(genquery).ATTRIBUTE(strict) -->
        If true, targets whose queries escape the transitive closure of their scopes will fail to
        build. If false, Bazel will print a warning and skip whatever query path led it outside of
        the scope, while completing the rest of the query.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("strict", BOOLEAN).value(true))
        /* <!-- #BLAZE_RULE(genquery).ATTRIBUTE(expression) -->
        The query to be executed. In contrast to the command line and other places in BUILD files,
        labels here are resolved relative to the root directory of the workspace. For example, the
        label <code>:b</code> in this attribute in the file <code>a/BUILD</code> will refer to the
        target <code>//:b</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("expression", STRING).mandatory())
        /* <!-- #BLAZE_RULE(genquery).ATTRIBUTE(opts) -->
        The options that are passed to the query engine. These correspond to the command line
        options that can be passed to <code>bazel query</code>. Some query options are not allowed
        here: <code>--keep_going</code>, <code>--query_file</code>, <code>--universe_scope</code>,
        <code>--order_results</code> and <code>--order_output</code>. Options not specified here
        will have their default values just like on the command line of <code>bazel query</code>.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("opts", STRING_LIST))
        /* <!-- #BLAZE_RULE(genquery).ATTRIBUTE(compressed_output) -->
        If <code>True</code>, query output is written in GZIP file format. This setting can be used
        to avoid spikes in Bazel's memory use when the query output is expected to be large. Bazel
        already internally compresses query outputs greater than 2<sup>20</sup> bytes regardless of
        the value of this setting, so setting this to <code>True</code> may not reduce retained
        heap. However, it allows Bazel to skip <em>decompression</em> when writing the output file,
        which can be memory-intensive.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("compressed_output", BOOLEAN).value(false))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("genquery")
        .ancestors(BaseRuleClasses.NativeActionCreatingRule.class)
        .factoryClass(GenQuery.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = genquery, FAMILY = General)[GENERIC_RULE] -->

  <p>
  <code>genquery()</code> runs a query specified in the
    <a href="${link query}">Blaze query language</a> and dumps the result
    into a file.
  </p>
  <p>
    In order to keep the build consistent, the query is allowed only to visit
    the transitive closure of the targets specified in the <code>scope</code>
    attribute. Queries violating this rule will fail during execution if
    <code>strict</code> is unspecified or true (if <code>strict</code> is false,
    the out of scope targets will simply be skipped with a warning). The
    easiest way to make sure this does not happen is to mention the same labels
    in the scope as in the query expression.
  </p>
  <p>
    The only difference between the queries allowed here and on the command
    line is that queries containing wildcard target specifications (e.g.
    <code>//pkg:*</code> or <code>//pkg:all</code>) are not allowed here.
    The reasons for this are two-fold: first, because <code>genquery</code> has
    to specify a scope to prevent targets outside the transitive closure of the
    query to influence its output; and, second, because <code>BUILD</code> files
    do not support wildcard dependencies (e.g. <code>deps=["//a/..."]</code>
    is not allowed).
  </p>
  <p>
    The genquery's output is ordered lexicographically in order to enforce deterministic output,
    with the exception of <code>--output=graph|minrank|maxrank</code> or when <code>somepath</code>
    is used as the top-level function.
  <p>
    The name of the output file is the name of the rule.
  </p>

<h4 id="genquery_examples">Examples</h4>
  <p>
    This example writes the list of the labels in the transitive closure of the
    specified target to a file.
  </p>

<pre class="code">
genquery(
    name = "kiwi-deps",
    expression = "deps(//kiwi:kiwi_lib)",
    scope = ["//kiwi:kiwi_lib"],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
