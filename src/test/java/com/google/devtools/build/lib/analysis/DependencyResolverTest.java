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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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

  @Before
  public final void createResolver() throws Exception {
    dependencyResolver = new DependencyResolver(ruleClassProvider.getDynamicTransitionMapper()) {

      @Override
      protected void invalidVisibilityReferenceHook(TargetAndConfiguration node, Label label) {
        throw new IllegalStateException();
      }

      @Override
      protected void invalidPackageGroupReferenceHook(TargetAndConfiguration node, Label label) {
        throw new IllegalStateException();
      }

      @Override
      protected void missingEdgeHook(Target from, Label to, NoSuchThingException e) {
        throw new IllegalStateException(e);
      }

      @Nullable
      @Override
      protected Target getTarget(Target from, Label label, NestedSetBuilder<Label> rootCauses) {
        try {
          return packageManager.getTarget(reporter, label);
        } catch (NoSuchPackageException | NoSuchTargetException | InterruptedException e) {
          throw new IllegalStateException(e);
        }
      }

      @Nullable
      @Override
      protected List<BuildConfiguration> getConfigurations(
          Set<Class<? extends BuildConfiguration.Fragment>> fragments,
          Iterable<BuildOptions> buildOptions) {
        throw new UnsupportedOperationException(
            "this functionality is covered by analysis-phase integration tests");
      }
    };
  }

  private void pkg(String name, String... contents) throws Exception {
    scratch.file("" + name + "/BUILD", contents);
  }

  private OrderedSetMultimap<Attribute, Dependency> dependentNodeMap(
      String targetName, NativeAspectClass aspect) throws Exception {
    Target target = packageManager.getTarget(reporter, Label.parseAbsolute(targetName));
    return dependencyResolver.dependentNodeMap(
        new TargetAndConfiguration(target, getTargetConfiguration()),
        getHostConfiguration(),
        aspect != null ? Aspect.forNative(aspect) : null,
        ImmutableMap.<Label, ConfigMatchingProvider>of(),
        /*toolchainContext=*/ null);
  }

  @SafeVarargs
  private final Dependency assertDep(
      OrderedSetMultimap<Attribute, Dependency> dependentNodeMap,
      String attrName,
      String dep,
      AspectDescriptor... aspects) {
    Attribute attr = null;
    for (Attribute candidate : dependentNodeMap.keySet()) {
      if (candidate.getName().equals(attrName)) {
        attr = candidate;
        break;
      }
    }

    assertWithMessage("Attribute '" + attrName + "' not found").that(attr).isNotNull();
    Dependency dependency = null;
    for (Dependency candidate : dependentNodeMap.get(attr)) {
      if (candidate.getLabel().toString().equals(dep)) {
        dependency = candidate;
        break;
      }
    }

    assertWithMessage("Dependency '" + dep + "' on attribute '" + attrName + "' not found")
        .that(dependency)
        .isNotNull();
    assertThat(dependency.getAspects().getAllAspects()).containsExactly((Object[]) aspects);
    return dependency;
  }

  @Test
  public void hasAspectsRequiredByRule() throws Exception {
    setRulesAvailableInTests(TestAspects.ASPECT_REQUIRING_RULE, TestAspects.BASE_RULE);
    pkg("a",
        "aspect(name='a', foo=[':b'])",
        "aspect(name='b', foo=[])");
    OrderedSetMultimap<Attribute, Dependency> map = dependentNodeMap("//a:a", null);
    assertDep(
        map, "foo", "//a:b",
        new AspectDescriptor(TestAspects.SIMPLE_ASPECT));
  }

  @Test
  public void hasAspectsRequiredByAspect() throws Exception {
    setRulesAvailableInTests(TestAspects.BASE_RULE, TestAspects.SIMPLE_RULE);
    pkg("a",
        "simple(name='a', foo=[':b'])",
        "simple(name='b', foo=[])");
    OrderedSetMultimap<Attribute, Dependency> map =
        dependentNodeMap("//a:a", TestAspects.ATTRIBUTE_ASPECT);
    assertDep(
        map, "foo", "//a:b",
        new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT));
  }

  @Test
  public void hasAllAttributesAspect() throws Exception {
    setRulesAvailableInTests(TestAspects.BASE_RULE, TestAspects.SIMPLE_RULE);
    pkg("a",
        "simple(name='a', foo=[':b'])",
        "simple(name='b', foo=[])");
    OrderedSetMultimap<Attribute, Dependency> map =
        dependentNodeMap("//a:a", TestAspects.ALL_ATTRIBUTES_ASPECT);
    assertDep(
        map, "foo", "//a:b",
        new AspectDescriptor(TestAspects.ALL_ATTRIBUTES_ASPECT));
  }

  @Test
  public void hasAspectDependencies() throws Exception {
    setRulesAvailableInTests(TestAspects.BASE_RULE);
    pkg("a", "base(name='a')");
    pkg("extra", "base(name='extra')");
    OrderedSetMultimap<Attribute, Dependency> map =
        dependentNodeMap("//a:a", TestAspects.EXTRA_ATTRIBUTE_ASPECT);
    assertDep(map, "$dep", "//extra:extra");
  }

  /**
   * Null configurations should always be explicit (vs. holding transitions). This lets Bazel skip
   * its complicated dependency configuration logic for these cases.
   */
  @Test
  public void nullConfigurationsAlwaysExplicit() throws Exception {
    pkg("a",
        "genrule(name = 'gen', srcs = ['gen.in'], cmd = '', outs = ['gen.out'])");
    update();
    Dependency dep = assertDep(dependentNodeMap("//a:gen", null), "srcs", "//a:gen.in");
    assertThat(dep.hasExplicitConfiguration()).isTrue();
    assertThat(dep.getConfiguration()).isNull();
  }

  /** Runs the same test with trimmed configurations. */
  @TestSpec(size = Suite.SMALL_TESTS)
  @RunWith(JUnit4.class)
  public static class WithTrimmedConfigurations extends DependencyResolverTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return super.defaultFlags().with(Flag.TRIMMED_CONFIGURATIONS);
    }
  }
}
