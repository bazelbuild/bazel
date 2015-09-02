// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.analysis.util.TestAspects.AspectRequiringRule;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for aspect creation and merging with configured targets.
 *
 * <p>Uses the complete analysis machinery and depends on custom rules so that behaviors related to
 * aspects can be tested even if they aren't used by regular rules.
 */
@RunWith(JUnit4.class)
public class AspectTest extends AnalysisTestCase {
  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
  }

  @Override
  @After
  public void tearDown() throws Exception {
    super.tearDown();
  }

  @SafeVarargs
  private final void setRules(RuleDefinition... rules) throws Exception {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    for (RuleDefinition rule : rules) {
      builder.addRuleDefinition(rule);
    }

    useRuleClassProvider(builder.build());
    update();
  }

  private void pkg(String name, String... contents) throws Exception {
    scratch.file("" + name + "/BUILD", contents);
  }

  @Test
  public void providersOfAspectAreMergedIntoDependency() throws Exception {
    setRules(new TestAspects.BaseRule(), new AspectRequiringRule());
    pkg("a",
        "aspect(name='a', foo=[':b'])",
        "aspect(name='b', foo=[])");

    ConfiguredTarget a = getConfiguredTarget("//a:a");
    assertThat(a.getProvider(TestAspects.RuleInfo.class).getData())
        .containsExactly("aspect //a:b", "rule //a:a");
  }

  @Test
  public void aspectIsNotCreatedIfAdvertisedProviderIsNotPresent() throws Exception {
    setRules(new TestAspects.BaseRule(), new TestAspects.LiarRule(),
        new TestAspects.AspectRequiringProviderRule());

    pkg("a",
        "aspect_requiring_provider(name='a', foo=[':b'])",
        "liar(name='b', foo=[])");

    ConfiguredTarget a = getConfiguredTarget("//a:a");
    assertThat(a.getProvider(TestAspects.RuleInfo.class).getData())
        .containsExactly("rule //a:a");
  }

  @Test
  public void aspectCreatedIfAdvertisedProviderIsPresent() throws Exception {
    setRules(new TestAspects.BaseRule(), new TestAspects.HonestRule(),
        new TestAspects.AspectRequiringProviderRule());

    pkg("a",
        "aspect_requiring_provider(name='a', foo=[':b'])",
        "honest(name='b', foo=[])");

    ConfiguredTarget a = getConfiguredTarget("//a:a");
    assertThat(a.getProvider(TestAspects.RuleInfo.class).getData())
        .containsExactly("rule //a:a", "aspect //a:b");
  }

  @Test
  public void aspectInError() throws Exception {
    setRules(new TestAspects.BaseRule(), new TestAspects.ErrorAspectRule(),
        new TestAspects.SimpleRule());

    pkg("a",
        "simple(name='a', foo=[':b'])",
        "error_aspect(name='b', foo=[':c'])",
        "simple(name='c')");

    reporter.removeHandler(failFastHandler);
    // getConfiguredTarget() uses a separate code path that does not hit
    // SkyframeBuildView#configureTargets
    try {
      update("//a:a");
      fail();
    } catch (ViewCreationFailedException e) {
      // expected
    }
    assertContainsEvent("Aspect error");
  }

  @Test
  public void sameTargetInDifferentAttributes() throws Exception {
    setRules(new TestAspects.BaseRule(), new TestAspects.AspectRequiringRule(),
        new TestAspects.SimpleRule());
    pkg("a",
        "aspect(name='a', foo=[':b'], bar=[':b'])",
        "aspect(name='b', foo=[])");

    ConfiguredTarget a = getConfiguredTarget("//a:a");
    assertThat(a.getProvider(TestAspects.RuleInfo.class).getData())
        .containsExactly("aspect //a:b", "rule //a:a");
  }

  @Test
  public void informationFromBaseRulePassedToAspect() throws Exception {
    setRules(new TestAspects.BaseRule(), new TestAspects.HonestRule(),
        new TestAspects.AspectRequiringProviderRule());

    pkg("a",
        "aspect_requiring_provider(name='a', foo=[':b'], baz='hello')",
        "honest(name='b', foo=[])");

    ConfiguredTarget a = getConfiguredTarget("//a:a");
    assertThat(a.getProvider(TestAspects.RuleInfo.class).getData())
        .containsExactly("rule //a:a", "aspect //a:b data hello");
  }
}
