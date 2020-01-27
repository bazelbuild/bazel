// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.test;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.common.options.Options;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Trimming transition factory which removes the test config fragment when entering a non-test rule.
 */
public final class TestTrimmingTransitionFactory implements TransitionFactory<Rule> {

  private static final Set<String> TEST_OPTIONS =
      ImmutableSet.copyOf(Options.getDefaults(TestOptions.class).asMap().keySet());

  /**
   * Trimming transition which removes the test config fragment if --trim_test_configuration is on.
   */
  public enum TestTrimmingTransition implements PatchTransition {
    INSTANCE;

    @Override
    public BuildOptions patch(BuildOptions originalOptions) {
      if (!originalOptions.contains(TestOptions.class)) {
        // nothing to do, already trimmed this fragment
        return originalOptions;
      }
      TestOptions originalTestOptions = originalOptions.get(TestOptions.class);
      if (!originalTestOptions.trimTestConfiguration) {
        // nothing to do, trimming is disabled
        return originalOptions;
      }
      return originalOptions.toBuilder().removeFragmentOptions(TestOptions.class).build();
    }
  }

  @Override
  public PatchTransition create(Rule rule) {
    RuleClass ruleClass = rule.getRuleClassObject();
    if (ruleClass
        .getConfigurationFragmentPolicy()
        .isLegalConfigurationFragment(TestConfiguration.class)) {
      // Test rule; no need to trim here.
      return NoTransition.INSTANCE;
    }

    Set<String> referencedTestOptions =
        new LinkedHashSet<String>(ruleClass.getOptionReferenceFunction().apply(rule));
    referencedTestOptions.retainAll(TEST_OPTIONS);
    if (!referencedTestOptions.isEmpty()) {
      // Test-option-referencing config_setting; no need to trim here.
      return NoTransition.INSTANCE;
    }

    // Non-test rule. Trim it!
    return TestTrimmingTransition.INSTANCE;
  }
}
