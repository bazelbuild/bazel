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
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.LateBoundLabel;
import com.google.devtools.build.lib.packages.AttributeMap;
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
  public static final class TestAspectClass extends NativeAspectClass
    implements ConfiguredAspectFactory {
    private AspectDefinition definition;

    public void setAspectDefinition(AspectDefinition definition) {
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

  public static final TestAspectClass TEST_ASPECT_CLASS = new TestAspectClass();

  @Test
  public void testAspectWithImplicitOrLateboundAttribute_AddsToAttributeMap() throws Exception {
    Attribute implicit = attr("$runtime", BuildType.LABEL)
        .value(Label.parseAbsoluteUnchecked("//run:time"))
        .build();
    LateBoundLabel<String> latebound = new LateBoundLabel<String>() {
        @Override
        public Label resolve(Rule rule, AttributeMap attributes, String configuration) {
          return Label.parseAbsoluteUnchecked("//run:away");
        }
    };
    AspectDefinition simple = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
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
      new AspectDefinition.Builder(TEST_ASPECT_CLASS)
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
      new AspectDefinition.Builder(TEST_ASPECT_CLASS)
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
    AspectDefinition withAspects = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .attributeAspect("srcs", TEST_ASPECT_CLASS)
        .attributeAspect("deps", TEST_ASPECT_CLASS)
        .build();

    assertThat(withAspects.getAttributeAspects(createLabelListAttribute("srcs")))
        .containsExactly(TEST_ASPECT_CLASS);
    assertThat(withAspects.getAttributeAspects(createLabelListAttribute("deps")))
        .containsExactly(TEST_ASPECT_CLASS);
  }

  @Test
  public void testAttributeAspect_AllAttributes() throws Exception {
    AspectDefinition withAspects = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .allAttributesAspect(TEST_ASPECT_CLASS)
        .build();

    assertThat(withAspects.getAttributeAspects(createLabelListAttribute("srcs")))
        .containsExactly(TEST_ASPECT_CLASS);
    assertThat(withAspects.getAttributeAspects(createLabelListAttribute("deps")))
        .containsExactly(TEST_ASPECT_CLASS);
  }


  private static Attribute createLabelListAttribute(String name) {
    return Attribute.attr(name, BuildType.LABEL_LIST)
        .allowedFileTypes(FileTypeSet.ANY_FILE)
        .build();
  }

  @Test
  public void testRequireProvider_AddsToSetOfRequiredProvidersAndNames() throws Exception {
    AspectDefinition requiresProviders = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .requireProviders(String.class, Integer.class)
        .build();
    assertThat(requiresProviders.getRequiredProviders()).hasSize(1);
    assertThat(requiresProviders.getRequiredProviders().get(0))
        .containsExactly(String.class, Integer.class);
    assertThat(requiresProviders.getRequiredProviderNames()).hasSize(1);
    assertThat(requiresProviders.getRequiredProviderNames().get(0))
        .containsExactly("java.lang.String", "java.lang.Integer");
  }

 @Test
  public void testRequireProvider_AddsTwoSetsOfRequiredProvidersAndNames() throws Exception {
    AspectDefinition requiresProviders = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .requireProviderSets(
            ImmutableList.of(
                ImmutableSet.<Class<?>>of(String.class, Integer.class),
                ImmutableSet.<Class<?>>of(Boolean.class)))
        .build();
    assertThat(requiresProviders.getRequiredProviders()).hasSize(2);
    assertThat(requiresProviders.getRequiredProviders().get(0))
        .containsExactly(String.class, Integer.class);
    assertThat(requiresProviders.getRequiredProviders().get(1))
        .containsExactly(Boolean.class);
    assertThat(requiresProviders.getRequiredProviderNames()).hasSize(2);
    assertThat(requiresProviders.getRequiredProviderNames().get(0))
        .containsExactly("java.lang.String", "java.lang.Integer");
    assertThat(requiresProviders.getRequiredProviderNames().get(1))
        .containsExactly("java.lang.Boolean");
  }

  @Test
  public void testNoConfigurationFragmentPolicySetup_HasNonNullPolicy() throws Exception {
    AspectDefinition noPolicy = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .build();
    assertThat(noPolicy.getConfigurationFragmentPolicy()).isNotNull();
  }

  @Test
  public void testMissingFragmentPolicy_PropagatedToConfigurationFragmentPolicy() throws Exception {
    AspectDefinition missingFragments = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .setMissingFragmentPolicy(MissingFragmentPolicy.IGNORE)
        .build();
    assertThat(missingFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(missingFragments.getConfigurationFragmentPolicy().getMissingFragmentPolicy())
        .isEqualTo(MissingFragmentPolicy.IGNORE);
  }

  @Test
  public void testRequiresConfigurationFragments_PropagatedToConfigurationFragmentPolicy()
      throws Exception {
    AspectDefinition requiresFragments = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
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
    AspectDefinition requiresFragments = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
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
    AspectDefinition requiresFragments = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
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
    AspectDefinition requiresFragments = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .requiresHostConfigurationFragmentsBySkylarkModuleName(ImmutableList.of("test_fragment"))
        .build();
    assertThat(requiresFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
        requiresFragments.getConfigurationFragmentPolicy()
            .isLegalConfigurationFragment(TestFragment.class, ConfigurationTransition.HOST))
        .isTrue();
  }

  @Test
  public void testEmptySkylarkConfigurationFragmentPolicySetup_HasNonNullPolicy() throws Exception {
    AspectDefinition noPolicy = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .requiresConfigurationFragmentsBySkylarkModuleName(ImmutableList.<String>of())
        .requiresHostConfigurationFragmentsBySkylarkModuleName(ImmutableList.<String>of())
        .build();
    assertThat(noPolicy.getConfigurationFragmentPolicy()).isNotNull();
  }

  @SkylarkModule(name = "test_fragment", doc = "test fragment")
  private static final class TestFragment {}
}
