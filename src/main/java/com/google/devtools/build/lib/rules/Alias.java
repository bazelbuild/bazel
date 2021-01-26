// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import static com.google.devtools.build.lib.packages.Attribute.ANY_RULE;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.VisibilityProvider;
import com.google.devtools.build.lib.analysis.VisibilityProviderImpl;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.util.FileTypeSet;

/**
 * Implementation of the <code>alias</code> rule.
 */
public class Alias implements RuleConfiguredTargetFactory {

  public static final String RULE_NAME = "alias";
  public static final String ACTUAL_ATTRIBUTE_NAME = "actual";

  @Override
  public ConfiguredTarget create(RuleContext ruleContext)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    ConfiguredTarget actual = (ConfiguredTarget) ruleContext.getPrerequisite("actual");

    // TODO(b/129045294): Remove once the flag is flipped.
    if (ruleContext.getLabel().getCanonicalForm().startsWith("@bazel_tools//platforms")
        && ruleContext
            .getConfiguration()
            .getOptions()
            .get(CoreOptions.class)
            .usePlatformsRepoForConstraints) {
      throw ruleContext.throwWithRuleError(
          "Constraints from @bazel_tools//platforms have been "
              + "removed. Please use constraints from @platforms repository embedded in "
              + "Bazel, or preferably declare dependency on "
              + "https://github.com/bazelbuild/platforms. See "
              + "https://github.com/bazelbuild/bazel/issues/8622 for details.");
    }

    return new AliasConfiguredTarget(
        ruleContext,
        actual,
        ImmutableMap.of(
            AliasProvider.class,
            AliasProvider.fromAliasRule(ruleContext.getLabel(), actual),
            VisibilityProvider.class,
            new VisibilityProviderImpl(ruleContext.getVisibility())));
  }

  /**
   * Rule definition.
   */
  public static class AliasRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          /*<!-- #BLAZE_RULE(alias).ATTRIBUTE(actual) -->
          The target this alias refers to. It does not need to be a rule, it can also be an input
          file.
          <!-- #END_BLAZE_RULE.ATTRIBUTE -->*/
          .removeAttribute("licenses")
          .removeAttribute("distribs")
          .add(
              attr(ACTUAL_ATTRIBUTE_NAME, LABEL)
                  .allowedFileTypes(FileTypeSet.ANY_FILE)
                  .allowedRuleClasses(ANY_RULE)
                  .mandatory())
          .canHaveAnyProvider()
          // Aliases themselves do not need toolchains or an execution platform, so this is fine.
          // The actual target will resolve platforms and toolchains with no issues regardless of
          // this setting.
          .useToolchainResolution(false)
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return Metadata.builder()
          .name(RULE_NAME)
          .factoryClass(Alias.class)
          .ancestors(BaseRuleClasses.NativeBuildRule.class)
          .build();
    }
  }
}

/*<!-- #BLAZE_RULE (NAME = alias, FAMILY = General)[GENERIC_RULE] -->

<p>
  The <code>alias</code> rule creates another name a rule can be referred to as.
</p>

<p>
  Aliasing only works for "regular" targets. In particular, <code>package_group</code>
  and <code>test_suite</code> cannot be aliased.
</p>

<p>
  The alias rule has its own visibility declaration. In all other respects, it behaves
  like the rule it references with some minor exceptions:

  <ul>
    <li>
      Tests are not run if their alias is mentioned on the command line. To define an alias
      that runs the referenced test, use a <a href="#test_suite"><code>test_suite</code></a>
      rule with a single target in its <a href="#test_suite.tests"><code>tests</code></a>
      attribute.
    </li>
    <li>
      When defining environment groups, the aliases to <code>environment</code> rules are not
      supported. They are not supported in the <code>--target_environment</code> command line
      option, either.
    </li>
  </ul>
</p>

<h4 id="alias_example">Examples</h4>

<pre class="code">
filegroup(
    name = "data",
    srcs = ["data.txt"],
)

alias(
    name = "other",
    actual = ":data",
)
</pre>

<!-- #END_BLAZE_RULE -->*/
