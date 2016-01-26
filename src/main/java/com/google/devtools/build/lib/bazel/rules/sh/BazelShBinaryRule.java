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

import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.BazelBaseRuleClasses;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShRuleClasses.ShRule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;

/**
 * Rule definition for the sh_binary rule.
 */
public final class BazelShBinaryRule implements RuleDefinition {
  @Override
  public RuleClass build(Builder builder, RuleDefinitionEnvironment environment) {
    return builder.build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("sh_binary")
        .ancestors(ShRule.class, BazelBaseRuleClasses.BinaryBaseRule.class)
        .factoryClass(ShBinary.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = sh_binary, TYPE = BINARY, FAMILY = Shell) -->

<p>
  The <code>sh_binary</code> rule is used to declare executable Bourne shell scripts.
  (<code>sh_binary</code> is a misnomer: its outputs aren't necessarily binaries.) This rule ensures
  that all dependencies are built, and appear in the <code>runfiles</code> area at execution time.
  We recommend that you name your <code>sh_binary()</code> rules after the name of the script minus
  the extension (e.g. <code>.sh</code>); do not give the rule and the file the same name.
</p>

<h4 id="sh_binary_examples">Example</h4>

<p>For a simple shell script with no dependencies or data:
</p>

<pre class="code">
sh_binary(
    name = "foo",
    srcs = ["foo.sh"],
    data = glob(["datafiles/*.txt"]),
)
</pre>

<!-- #END_BLAZE_RULE -->*/
