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
import com.google.devtools.build.lib.analysis.config.BuildOptionsCache;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.common.options.Options;

/**
 * Trimming transition factory which removes the test config fragment when entering a non-test rule.
 */
public final class TestTrimmingTransitionFactory implements TransitionFactory<Rule> {

  private static final ImmutableSet<String> TEST_OPTIONS =
      ImmutableSet.copyOf(Options.getDefaults(TestOptions.class).asMap().keySet());

  /**
   * Trimming transition which removes the test config fragment if --trim_test_configuration is on.
   */
  public enum TestTrimmingTransition implements PatchTransition {
    INSTANCE;

    // This cache is to prevent major slowdowns when using --trim_test_configuration. This
    // transition is always invoked on every target in the top-level invocation. Thus, a wide
    // invocation, like //..., will cause the transition to be invoked on a large number of targets
    // leading to significant performance degradation. (Notably, the transition itself is somewhat
    // fast; however, the post-processing of the BuildOptions results into a BuildConfiguration
    // takes a significant amount of time).
    private static final BuildOptionsCache<Integer> cache = new BuildOptionsCache<>();

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(TestOptions.class);
    }

    @Override
    public BuildOptions patch(BuildOptionsView originalOptions, EventHandler eventHandler) {
      if (!originalOptions.contains(TestOptions.class)) {
        // nothing to do, already trimmed this fragment
        return originalOptions.underlying();
      }
      TestOptions originalTestOptions = originalOptions.get(TestOptions.class);
      if (!originalTestOptions.trimTestConfiguration) {
        // nothing to do, trimming is disabled
        return originalOptions.underlying();
      }
      return cache.applyTransition(
          originalOptions,
          // The transition uses no non-BuildOptions arguments
          0,
          () ->
              originalOptions.underlying().toBuilder()
                  .removeFragmentOptions(TestOptions.class)
                  .build());
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

    for (String referencedOptions : ruleClass.getOptionReferenceFunction().apply(rule)) {
      if (TEST_OPTIONS.contains(referencedOptions)) {
        // Test-option-referencing config_setting; no need to trim here.
        return NoTransition.INSTANCE;
      }
    }

    // Non-test rule. Trim it!
    return TestTrimmingTransition.INSTANCE;
  }
}
