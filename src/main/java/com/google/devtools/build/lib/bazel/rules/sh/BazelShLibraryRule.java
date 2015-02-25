// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.sh;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.Type.LABEL_LIST;

import com.google.devtools.build.lib.analysis.BlazeRule;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShRuleClasses.ShRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for the sh_library rule.
 */
@BlazeRule(name = "sh_library",
             ancestors = { ShRule.class },
             factoryClass = ShLibrary.class)
public final class BazelShLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(sh_library).ATTRIBUTE(deps) -->
        The list of other targets to be aggregated in to this "library" target.
        <i>(List of <a href="build-ref.html#labels">labels</a>; optional)</i><br/>
        See general comments about <code>deps</code>
        at <a href="#common-attributes">Attributes common to all build rules</a>.
        You should use this attribute to list other
        <code>sh_library</code> or <code>proto_library</code> rules that provide
        interpreted program source code depended on by the code in
        <code>srcs</code>.  If you depend on a <code>proto_library</code> target,
        the proto sources in that target will be included in this library, but
        no generated files will be built.
        <!-- #END_BLAZE_RULE.ATTRIBUTE --> */

        /* <!-- #BLAZE_RULE(sh_library).ATTRIBUTE(srcs) -->
        The list of input files.
        <i>(List of <a href="build-ref.html#labels">labels</a>,
        optional)</i><br/>
        You should use this attribute to list interpreted program
        source files that belong to this package, such as additional
        files containing Bourne shell subroutines, loaded via the shell's
        <code>source</code> or <code>.</code> command.
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .override(attr("srcs", LABEL_LIST).legacyAllowAnyFileType())
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = sh_library, TYPE = LIBRARY, FAMILY = Shell) -->

${ATTRIBUTE_SIGNATURE}

<p>
  The main use for this rule is to aggregate together a logical
  "library" consisting of related scripts&mdash;programs in an
  interpreted language that does not require compilation or linking,
  such as the Bourne shell&mdash;and any data those programs need at
  run-time.  Such "libraries" can then be used from
  the <code>data</code> attribute of one or
  more <code>sh_binary</code> rules.
</p>

<p>
  Historically, a second use was to aggregate a collection of data files
  together, to ensure that they are available at runtime in
  the <code>.runfiles</code> area of one or more <code>*_binary</code>
  rules (not necessarily <code>sh_binary</code>).
  However, the <a href="#filegroup"><code>filegroup()</code></a> rule
  should be used now; it is intended to replace this use of
  <code>sh_library</code>.
</p>

<p>
  In interpreted programming languages, there's not always a clear
  distinction between "code" and "data": after all, the program is
  just "data" from the interpreter's point of view.  For this reason
  (and historical accident) this rule has three attributes which are
  all essentially equivalent: <code>srcs</code>, <code>deps</code>
  and <code>data</code>.
  The recommended usage of each attribute is mentioned below.  The
  current implementation does not distinguish the elements of these lists.
  All three attributes accept rules, source files and derived files.
</p>

${ATTRIBUTE_DEFINITION}

<h4 id="sh_library_examples">Examples</h4>

<pre class="code">
sh_library(
    name = "foo",
    data = [
        ":foo_service_script",  # a sh_binary with srcs
        ":deploy_foo",  # another sh_binary with srcs
    ],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
