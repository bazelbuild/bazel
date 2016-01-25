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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.syntax.Type.BOOLEAN;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static com.google.devtools.build.lib.syntax.Type.STRING_LIST;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for genquery the rule.
 */
public final class GenQueryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
    return builder
        /* <!-- #BLAZE_RULE(genquery).ATTRIBUTE(scope) -->
        The scope of the query. The query is not allowed to
        touch targets outside the transitive closure of these targets.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("scope", LABEL_LIST).mandatory().legacyAllowAnyFileType())
        /* <!-- #BLAZE_RULE(genquery).ATTRIBUTE(strict) -->
        If true, targets whose queries escape the transitive closure of their
        scopes will fail to build. If false, Blaze will print a warning and skip
        whatever query path led it outside of the scope, while completing the rest
        of the query.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("strict", BOOLEAN).value(true))
        /* <!-- #BLAZE_RULE(genquery).ATTRIBUTE(expression) -->
        The query to be executed.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("expression", STRING).mandatory())
        /* <!-- #BLAZE_RULE(genquery).ATTRIBUTE(opts) -->
        The options that are passed to the query engine.
        These correspond to the command line options that can be passed to
        <code>blaze query</code>. The only query options that are not allowed here
        are <code>--keep_going</code> and <code>--order_output</code>.
        ${SYNOPSIS}
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */
        .add(attr("opts", STRING_LIST))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("genquery")
        .ancestors(BaseRuleClasses.RuleBase.class)
        .factoryClass(GenQuery.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = genquery, TYPE = LIBRARY, FAMILY = General)[GENERIC_RULE] -->

  <p>
  <code>genquery()</code> runs a query specified in the
    <a href="../blaze-query-v2.html">Blaze query language</a> and dumps the result
    into a file.
  </p>
  <p>
    In order to keep the build consistent, the query is allowed only to visit
    the transitive closure of the targets specified in the <code>scope</code>
    attribute. Queries violating this rule will fail during execution if
    <code>strict</code> is unspecified or true (if <code>strict</code> is false,
    the out of scope targets will simply be skipped with a warning). The
    easiest way to make sure this does not happen is to mention the same labels
    in the scope as in the query expression. The only difference between the
    queries allowed here and on the command line is that queries containing
    wildcard target specifications (e.g. <code>//pkg:*</code> or
    <code>//pkg:all</code>) are not allowed here.
  </p>
  <p>
    The genquery's output is ordered using <code>--order_output=full</code> in
    order to enforce deterministic output.
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
