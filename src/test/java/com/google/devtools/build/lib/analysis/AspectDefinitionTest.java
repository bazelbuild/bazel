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
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.FileTypeSet;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for aspect definitions.
 */
@RunWith(JUnit4.class)
public class AspectDefinitionTest {
  /**
   * A dummy aspect factory. Is there to demonstrate how to define aspects and so that we can test
   * {@code attributeAspect}.
   */
  public static final class TestAspectFactory implements ConfiguredNativeAspectFactory {
    private final AspectDefinition definition;

    /**
     * Normal aspects will have an argumentless constructor and their definition will be hard-wired
     * as a static member. This one is different so that we can create the definition in a test
     * method.
     */
    private TestAspectFactory(AspectDefinition definition) {
      this.definition = definition;
    }

    @Override
    public ConfiguredAspect create(
        ConfiguredTarget base, RuleContext context, AspectParameters parameters) {
      throw new IllegalStateException();
    }

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return definition;
    }
  }

  @Test
  public void testAspectWithImplicitOrLateboundAttribute_AddsToAttributeMap() throws Exception {
    Attribute implicit = attr("$runtime", BuildType.LABEL)
        .value(Label.parseAbsoluteUnchecked("//run:time"))
        .build();
    LateBoundLabel<String> latebound = new LateBoundLabel<String>() {
        @Override
        public Label getDefault(Rule rule, String configuration) {
          return Label.parseAbsoluteUnchecked("//run:away");
        }
    };
    AspectDefinition simple = new AspectDefinition.Builder("simple")
        .add(implicit)
        .add(attr(":latebound", BuildType.LABEL).value(latebound))
        .build();
    assertThat(simple.getAttributes()).containsEntry("$runtime", implicit);
    assertThat(simple.getAttributes()).containsKey(":latebound");
    assertThat(simple.getAttributes().get(":latebound").getLateBoundDefault())
        .isEqualTo(latebound);
  }

  @Test
  public void testAspectWithDuplicateAttribute_FailsToAdd() throws Exception {
    try {
      new AspectDefinition.Builder("clash")
          .add(attr("$runtime", BuildType.LABEL).value(Label.parseAbsoluteUnchecked("//run:time")))
          .add(attr("$runtime", BuildType.LABEL).value(Label.parseAbsoluteUnchecked("//oops")));
      fail(); // expected IllegalArgumentException
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void testAspectWithUserVisibleAttribute_FailsToAdd() throws Exception {
    try {
      new AspectDefinition.Builder("user_visible_attribute")
          .add(
              attr("invalid", BuildType.LABEL)
                  .value(Label.parseAbsoluteUnchecked("//run:time"))
                  .allowedFileTypes(FileTypeSet.NO_FILE))
          .build();
      fail(); // expected IllegalArgumentException
    } catch (IllegalArgumentException e) {
      // expected
    }
  }

  @Test
  public void testAttributeAspect_WrapsAndAddsToMap() throws Exception {
    AspectDefinition withAspects = new AspectDefinition.Builder("attribute_aspect")
        .attributeAspect("srcs", TestAspectFactory.class)
        .attributeAspect("deps", new NativeAspectClass<TestAspectFactory>(TestAspectFactory.class))
        .build();
    assertThat(withAspects.getAttributeAspects())
        .containsEntry("srcs", new NativeAspectClass<TestAspectFactory>(TestAspectFactory.class));
    assertThat(withAspects.getAttributeAspects())
        .containsEntry("deps", new NativeAspectClass<TestAspectFactory>(TestAspectFactory.class));
  }

  @Test
  public void testRequireProvider_AddsToSetOfRequiredProvidersAndNames() throws Exception {
    AspectDefinition requiresProviders = new AspectDefinition.Builder("required_providers")
        .requireProvider(String.class)
        .requireProvider(Integer.class)
        .build();
    assertThat(requiresProviders.getRequiredProviders())
        .containsExactly(String.class, Integer.class);
    assertThat(requiresProviders.getRequiredProviderNames())
        .containsExactly("java.lang.String", "java.lang.Integer");
  }

  @Test
  public void testNoConfigurationFragmentPolicySetup_ReturnsNull() throws Exception {
    AspectDefinition noPolicy = new AspectDefinition.Builder("no_policy")
        .build();
    assertThat(noPolicy.getConfigurationFragmentPolicy()).isNull();
  }

  @Test
  public void testMissingFragmentPolicy_PropagatedToConfigurationFragmentPolicy() throws Exception {
    AspectDefinition missingFragments = new AspectDefinition.Builder("missing_fragments")
        .setMissingFragmentPolicy(MissingFragmentPolicy.IGNORE)
        .build();
    assertThat(missingFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(missingFragments.getConfigurationFragmentPolicy().getMissingFragmentPolicy())
        .isEqualTo(MissingFragmentPolicy.IGNORE);
  }

  @Test
  public void testRequiresConfigurationFragments_PropagatedToConfigurationFragmentPolicy()
      throws Exception {
    AspectDefinition requiresFragments = new AspectDefinition.Builder("requires_fragments")
        .requiresConfigurationFragments(Integer.class, String.class)
        .build();
    assertThat(requiresFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
        requiresFragments.getConfigurationFragmentPolicy().getRequiredConfigurationFragments())
            .containsExactly(Integer.class, String.class);
  }

  @Test
  public void testRequiresHostConfigurationFragments_PropagatedToConfigurationFragmentPolicy()
      throws Exception {
    AspectDefinition requiresFragments = new AspectDefinition.Builder("requires_fragments")
        .requiresHostConfigurationFragments(Integer.class, String.class)
        .build();
    assertThat(requiresFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
        requiresFragments.getConfigurationFragmentPolicy().getRequiredConfigurationFragments())
            .containsExactly(Integer.class, String.class);
  }

  @Test
  public void testRequiresConfigurationFragmentNames_PropagatedToConfigurationFragmentPolicy()
      throws Exception {
    AspectDefinition requiresFragments = new AspectDefinition.Builder("requires_fragments")
        .requiresConfigurationFragmentsBySkylarkModuleName(ImmutableList.of("test_fragment"))
        .build();
    assertThat(requiresFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
        requiresFragments.getConfigurationFragmentPolicy()
            .isLegalConfigurationFragment(TestFragment.class, ConfigurationTransition.NONE))
        .isTrue();
  }

  @Test
  public void testRequiresHostConfigurationFragmentNames_PropagatedToConfigurationFragmentPolicy()
      throws Exception {
    AspectDefinition requiresFragments = new AspectDefinition.Builder("requires_fragments")
        .requiresHostConfigurationFragmentsBySkylarkModuleName(ImmutableList.of("test_fragment"))
        .build();
    assertThat(requiresFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
        requiresFragments.getConfigurationFragmentPolicy()
            .isLegalConfigurationFragment(TestFragment.class, ConfigurationTransition.HOST))
        .isTrue();
  }

  @Test
  public void testEmptySkylarkConfigurationFragmentPolicySetup_ReturnsNull() throws Exception {
    AspectDefinition noPolicy = new AspectDefinition.Builder("no_policy")
        .requiresConfigurationFragmentsBySkylarkModuleName(ImmutableList.<String>of())
        .requiresHostConfigurationFragmentsBySkylarkModuleName(ImmutableList.<String>of())
        .build();
    assertThat(noPolicy.getConfigurationFragmentPolicy()).isNull();
  }

  @SkylarkModule(name = "test_fragment", doc = "test fragment")
  private static final class TestFragment {}
}
