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
package com.google.devtools.build.lib.bazel.rules.sh;

import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShRuleClasses.ShRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for the sh_library rule.
 */
public final class BazelShLibraryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder
        /* <!-- #BLAZE_RULE(sh_library).ATTRIBUTE(srcs) -->
        The list of input files.
        ${SYNOPSIS}
        <p>
          This attribute should be used to list shell script source files that belong to
          this library. Scripts can load other scripts using the shell's <code>source</code>
          or <code>.</code> command.
        </p>
        <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
        .override(attr("srcs", LABEL_LIST).allowedFileTypes(BazelShRuleClasses.SH_FILES))
        .build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("sh_library")
        .ancestors(ShRule.class)
        .factoryClass(ShLibrary.class)
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
  run-time. Such "libraries" can then be used from
  the <code>data</code> attribute of one or
  more <code>sh_binary</code> rules.
</p>

<p>
  You can use the <a href="general.html#filegroup"><code>filegroup</code></a> rule to aggregate data
  files.
</p>

<p>
  In interpreted programming languages, there's not always a clear
  distinction between "code" and "data": after all, the program is
  just "data" from the interpreter's point of view. For this reason
  this rule has three attributes which are all essentially equivalent:
  <code>srcs</code>, <code>deps</code> and <code>data</code>.
  The current implementation does not distinguish between the elements of these lists.
  All three attributes accept rules, source files and generated files.
  It is however good practice to use the attributes for their usual purpose (as with other rules).
</p>

${ATTRIBUTE_DEFINITION}

<h4 id="sh_library_examples">Examples</h4>

<pre class="code">
sh_library(
    name = "foo",
    data = [
        ":foo_service_script",  # an sh_binary with srcs
        ":deploy_foo",  # another sh_binary with srcs
    ],
)
</pre>

<!-- #END_BLAZE_RULE -->*/
