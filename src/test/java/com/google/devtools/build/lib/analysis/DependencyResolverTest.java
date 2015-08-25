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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.common.testing.EqualsTester;
import com.google.common.testing.NullPointerTester;
import com.google.devtools.build.lib.analysis.DependencyResolver.Dependency;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.analysis.util.TestAspects.AspectRequiringRule;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectFactory;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import javax.annotation.Nullable;

/**
 * Tests for {@link DependencyResolver}.
 *
 * <p>These use custom rules so that all usual and unusual cases related to aspect processing can
 * be tested.
 *
 * <p>It would be nicer is we didn't have a Skyframe executor, if we didn't have that, we'd need a
 * way to create a configuration, a package manager and a whole lot of other things, so it's just
 * easier this way.
 */
@RunWith(JUnit4.class)
public class DependencyResolverTest extends AnalysisTestCase {
  private DependencyResolver dependencyResolver;

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();

    dependencyResolver = new DependencyResolver() {
      @Override
      protected void invalidVisibilityReferenceHook(TargetAndConfiguration node, Label label) {
        throw new IllegalStateException();
      }

      @Override
      protected void invalidPackageGroupReferenceHook(TargetAndConfiguration node, Label label) {
        throw new IllegalStateException();
      }

      @Nullable
      @Override
      protected Target getTarget(Label label) throws NoSuchThingException {
        try {
          return packageManager.getTarget(reporter, label);
        } catch (InterruptedException e) {
          throw new IllegalStateException(e);
        }
      }
    };
  }

  @Override
  @After
  public void tearDown() throws Exception {
    super.tearDown();
  }

  private void pkg(String name, String... contents) throws Exception {
    scratch.file("" + name + "/BUILD", contents);
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

  private ListMultimap<Attribute, Dependency> dependentNodeMap(
      String targetName, Class<? extends ConfiguredAspectFactory> aspect) throws Exception {
    AspectDefinition aspectDefinition = aspect == null
        ? null
        : AspectFactory.Util.create(aspect).getDefinition();
    Target target = packageManager.getTarget(reporter, Label.parseAbsolute(targetName));
    return dependencyResolver.dependentNodeMap(
        new TargetAndConfiguration(target, getTargetConfiguration()),
        getHostConfiguration(),
        aspectDefinition,
        ImmutableSet.<ConfigMatchingProvider>of());
  }

  @SafeVarargs
  private final void assertDep(
      ListMultimap<Attribute, Dependency> dependentNodeMap,
      String attrName,
      String dep,
      Class<? extends AspectFactory<?, ?, ?>>... aspects) {
    Attribute attr = null;
    for (Attribute candidate : dependentNodeMap.keySet()) {
      if (candidate.getName().equals(attrName)) {
        attr = candidate;
        break;
      }
    }

    assertNotNull("Attribute '" + attrName + "' not found", attr);
    Dependency dependency = null;
    for (Dependency candidate : dependentNodeMap.get(attr)) {
      if (candidate.getLabel().toString().equals(dep)) {
        dependency = candidate;
        break;
      }
    }

    assertNotNull("Dependency '" + dep + "' on attribute '" + attrName + "' not found", dependency);
    assertThat(dependency.getAspects()).containsExactly((Object[]) aspects);
  }

  @Test
  public void hasAspectsRequiredByRule() throws Exception {
    setRules(new AspectRequiringRule(), new TestAspects.BaseRule());
    pkg("a",
        "aspect(name='a', foo=[':b'])",
        "aspect(name='b', foo=[])");
    ListMultimap<Attribute, Dependency> map = dependentNodeMap("//a:a", null);
    assertDep(map, "foo", "//a:b", TestAspects.SimpleAspect.class);
  }

  @Test
  public void hasAspectsRequiredByAspect() throws Exception {
    setRules(new TestAspects.BaseRule(), new TestAspects.SimpleRule());
    pkg("a",
        "simple(name='a', foo=[':b'])",
        "simple(name='b', foo=[])");
    ListMultimap<Attribute, Dependency> map =
        dependentNodeMap("//a:a", TestAspects.AttributeAspect.class);
    assertDep(map, "foo", "//a:b", TestAspects.AttributeAspect.class);
  }

  @Test
  public void hasAspectDependencies() throws Exception {
    setRules(new TestAspects.BaseRule());
    pkg("a", "base(name='a')");
    pkg("extra", "base(name='extra')");
    ListMultimap<Attribute, Dependency> map =
        dependentNodeMap("//a:a", TestAspects.ExtraAttributeAspect.class);
    assertDep(map, "$dep", "//extra:extra");
  }

  @Test
  public void constructorsForDependencyPassNullableTester() throws Exception {
    update();

    new NullPointerTester()
        .setDefault(Label.class, Label.parseAbsolute("//a"))
        .setDefault(BuildConfiguration.class, getTargetConfiguration())
        .setDefault(ImmutableSet.class, ImmutableSet.of())
        .testAllPublicConstructors(Dependency.class);
  }

  @Test
  public void equalsOnDependencyPassesEqualsTester() throws Exception {
    update();

    Label a = Label.parseAbsolute("//a");
    Label aExplicit = Label.parseAbsolute("//a:a");
    Label b = Label.parseAbsolute("//b");

    BuildConfiguration host = getHostConfiguration();
    BuildConfiguration target = getTargetConfiguration();

    ImmutableSet<Class<? extends ConfiguredAspectFactory>> twoAspects =
        ImmutableSet.<Class<? extends ConfiguredAspectFactory>>of(
            TestAspects.SimpleAspect.class, TestAspects.AttributeAspect.class);
    ImmutableSet<Class<? extends ConfiguredAspectFactory>> inverseAspects =
        ImmutableSet.<Class<? extends ConfiguredAspectFactory>>of(
            TestAspects.AttributeAspect.class, TestAspects.SimpleAspect.class);
    ImmutableSet<Class<? extends ConfiguredAspectFactory>> differentAspects =
        ImmutableSet.<Class<? extends ConfiguredAspectFactory>>of(
            TestAspects.AttributeAspect.class, TestAspects.ErrorAspect.class);

    new EqualsTester()
        .addEqualityGroup(
            // base set: //a, host configuration, normal aspect set
            new Dependency(a, host, twoAspects),
            new Dependency(aExplicit, host, twoAspects),
            new Dependency(a, host, inverseAspects),
            new Dependency(aExplicit, host, inverseAspects))
        .addEqualityGroup(
            // base set but with label //b
            new Dependency(b, host, twoAspects),
            new Dependency(b, host, inverseAspects))
        .addEqualityGroup(
            // base set but with target configuration
            new Dependency(a, target, twoAspects),
            new Dependency(aExplicit, target, twoAspects),
            new Dependency(a, target, inverseAspects),
            new Dependency(aExplicit, target, inverseAspects))
        .addEqualityGroup(
            // base set but with null configuration
            new Dependency(a, (BuildConfiguration) null, twoAspects),
            new Dependency(aExplicit, (BuildConfiguration) null, twoAspects),
            new Dependency(a, (BuildConfiguration) null, inverseAspects),
            new Dependency(aExplicit, (BuildConfiguration) null, inverseAspects))
        .addEqualityGroup(
            // base set but with different aspects
            new Dependency(a, host, differentAspects),
            new Dependency(aExplicit, host, differentAspects))
        .addEqualityGroup(
            // base set but with label //b and target configuration
            new Dependency(b, target, twoAspects),
            new Dependency(b, target, inverseAspects))
        .addEqualityGroup(
            // base set but with label //b and null configuration
            new Dependency(b, (BuildConfiguration) null, twoAspects),
            new Dependency(b, (BuildConfiguration) null, inverseAspects))
        .addEqualityGroup(
            // base set but with label //b and different aspects
            new Dependency(b, host, differentAspects))
        .addEqualityGroup(
            // base set but with target configuration and different aspects
            new Dependency(a, target, differentAspects),
            new Dependency(aExplicit, target, differentAspects))
        .addEqualityGroup(
            // base set but with null configuration and different aspects
            new Dependency(a, (BuildConfiguration) null, differentAspects),
            new Dependency(aExplicit, (BuildConfiguration) null, differentAspects))
        .addEqualityGroup(
            // inverse of base set: //b, target configuration, different aspects
            new Dependency(b, target, differentAspects))
        .addEqualityGroup(
            // inverse of base set: //b, null configuration, different aspects
            new Dependency(b, (BuildConfiguration) null, differentAspects))
        .testEquals();
  }
}
