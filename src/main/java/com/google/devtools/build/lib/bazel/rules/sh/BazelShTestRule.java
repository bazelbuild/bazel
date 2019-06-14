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
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.bazel.rules.sh.BazelShRuleClasses.ShRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;

/**
 * Rule definition for the sh_test rule.
 */
public final class BazelShTestRule implements RuleDefinition {
  @Override
  public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
    // TODO(bazel-team): Add $lcov_merger to every test rule as opposed to particular rules.
    builder
        .add(
            attr("$lcov_merger", LABEL)
                .value(
                    Label.parseAbsoluteUnchecked(
                        "@bazel_tools//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Main")))
        .add(
            attr("$launcher", LABEL)
                .cfg(HostTransition.createFactory())
                .value(environment.getToolsLabel("//tools/launcher:launcher")))
        // Add the script as an attribute in order for sh_test to output code coverage results for
        // code covered by CC binaries invocations.
        .add(
            attr("$collect_cc_coverage", LABEL)
                .cfg(HostTransition.createFactory())
                .singleArtifact()
                .value(environment.getToolsLabel("//tools/test:collect_cc_coverage")));
    return builder.build();
  }

  @Override
  public Metadata getMetadata() {
    return RuleDefinition.Metadata.builder()
        .name("sh_test")
        .type(RuleClassType.TEST)
        .ancestors(ShRule.class, BaseRuleClasses.TestBaseRule.class)
        .factoryClass(ShBinary.class)
        .build();
  }
}

/*<!-- #BLAZE_RULE (NAME = sh_test, TYPE = TEST, FAMILY = Shell) -->

<p>A <code>sh_test()</code> rule creates a test written as a Bourne shell script.</p>

<p>See the <a href="${link common-definitions#common-attributes-tests}">
attributes common to all test rules (*_test)</a>.</p>

<h4 id="sh_test_examples">Examples</h4>

<pre class="code">
sh_test(
    name = "foo_integration_test",
    size = "small",
    srcs = ["foo_integration_test.sh"],
    deps = [":foo_sh_lib"],
    data = glob(["testdata/*.txt"]),
)
</pre>

<!-- #END_BLAZE_RULE -->*/
