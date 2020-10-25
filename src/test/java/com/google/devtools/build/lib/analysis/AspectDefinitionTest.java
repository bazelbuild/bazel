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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.ConfigAwareAspectBuilder;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.AspectDefinition;
import com.google.devtools.build.lib.packages.AspectParameters;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.LabelLateBoundDefault;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.NativeAspectClass;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.util.FileTypeSet;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.StarlarkValue;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for aspect definitions.
 */
@RunWith(JUnit4.class)
public class AspectDefinitionTest {

  private static final class P1 implements TransitiveInfoProvider {}

  private static final class P2 implements TransitiveInfoProvider {}

  private static final class P3 implements TransitiveInfoProvider {}

  private static final class P4 implements TransitiveInfoProvider {}

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
        ConfiguredTargetAndData ctadBase,
        RuleContext context,
        AspectParameters parameters,
        String toolsRepository) {
      throw new IllegalStateException();
    }

    @Override
    public AspectDefinition getDefinition(AspectParameters aspectParameters) {
      return definition;
    }
  }

  public static final TestAspectClass TEST_ASPECT_CLASS = new TestAspectClass();

  @Test
  public void testAspectWithImplicitOrLateboundAttribute_addsToAttributeMap() throws Exception {
    Attribute implicit = attr("$runtime", BuildType.LABEL)
        .value(Label.parseAbsoluteUnchecked("//run:time"))
        .build();
    LabelLateBoundDefault<Void> latebound =
        LateBoundDefault.fromConstantForTesting(Label.parseAbsoluteUnchecked("//run:away"));
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
  public void testAspectWithDuplicateAttribute_failsToAdd() throws Exception {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new AspectDefinition.Builder(TEST_ASPECT_CLASS)
                .add(
                    attr("$runtime", BuildType.LABEL)
                        .value(Label.parseAbsoluteUnchecked("//run:time")))
                .add(
                    attr("$runtime", BuildType.LABEL)
                        .value(Label.parseAbsoluteUnchecked("//oops"))));
  }

  @Test
  public void testAspectWithUserVisibleAttribute_failsToAdd() throws Exception {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new AspectDefinition.Builder(TEST_ASPECT_CLASS)
                .add(
                    attr("invalid", BuildType.LABEL)
                        .value(Label.parseAbsoluteUnchecked("//run:time"))
                        .allowedFileTypes(FileTypeSet.NO_FILE))
                .build());
  }

  @Test
  public void testAttributeAspect_wrapsAndAddsToMap() throws Exception {
    AspectDefinition withAspects = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .propagateAlongAttribute("srcs")
        .propagateAlongAttribute("deps")
        .build();

    assertThat(withAspects.propagateAlong("srcs")).isTrue();
    assertThat(withAspects.propagateAlong("deps")).isTrue();
  }

  @Test
  public void testAttributeAspect_allAttributes() throws Exception {
    AspectDefinition withAspects = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .propagateAlongAllAttributes()
        .build();

    assertThat(withAspects.propagateAlong("srcs")).isTrue();
    assertThat(withAspects.propagateAlong("deps")).isTrue();
  }

  @Test
  public void testRequireProvider_addsToSetOfRequiredProvidersAndNames() throws Exception {
    AspectDefinition requiresProviders =
        new AspectDefinition.Builder(TEST_ASPECT_CLASS)
            .requireProviders(P1.class, P2.class)
            .build();
    AdvertisedProviderSet expectedOkSet =
        AdvertisedProviderSet.builder()
            .addNative(P1.class)
            .addNative(P2.class)
            .addNative(P3.class)
            .build();
    assertThat(requiresProviders.getRequiredProviders().isSatisfiedBy(expectedOkSet))
        .isTrue();

    AdvertisedProviderSet expectedFailSet =
        AdvertisedProviderSet.builder().addNative(P1.class).build();
    assertThat(requiresProviders.getRequiredProviders().isSatisfiedBy(expectedFailSet))
        .isFalse();

    assertThat(requiresProviders.getRequiredProviders().isSatisfiedBy(AdvertisedProviderSet.ANY))
        .isTrue();
    assertThat(requiresProviders.getRequiredProviders().isSatisfiedBy(AdvertisedProviderSet.EMPTY))
        .isFalse();
  }

  @Test
  public void testRequireProvider_addsTwoSetsOfRequiredProvidersAndNames() throws Exception {
    AspectDefinition requiresProviders =
        new AspectDefinition.Builder(TEST_ASPECT_CLASS)
            .requireProviderSets(
                ImmutableList.of(ImmutableSet.of(P1.class, P2.class), ImmutableSet.of(P3.class)))
            .build();

    AdvertisedProviderSet expectedOkSet1 =
        AdvertisedProviderSet.builder().addNative(P1.class).addNative(P2.class).build();

    AdvertisedProviderSet expectedOkSet2 =
        AdvertisedProviderSet.builder().addNative(P3.class).build();

    AdvertisedProviderSet expectedFailSet =
        AdvertisedProviderSet.builder().addNative(P4.class).build();

   assertThat(requiresProviders.getRequiredProviders().isSatisfiedBy(AdvertisedProviderSet.ANY))
       .isTrue();
    assertThat(requiresProviders.getRequiredProviders().isSatisfiedBy(expectedOkSet1)).isTrue();
    assertThat(requiresProviders.getRequiredProviders().isSatisfiedBy(expectedOkSet2)).isTrue();
    assertThat(requiresProviders.getRequiredProviders().isSatisfiedBy(expectedFailSet)).isFalse();
   assertThat(requiresProviders.getRequiredProviders().isSatisfiedBy(AdvertisedProviderSet.EMPTY))
       .isFalse();

 }

  @Test
  public void testRequireAspectClass_defaultAcceptsNothing() {
    AspectDefinition noAspects = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .build();

    AdvertisedProviderSet expectedFailSet =
        AdvertisedProviderSet.builder().addNative(P4.class).build();

    assertThat(noAspects.getRequiredProvidersForAspects().isSatisfiedBy(AdvertisedProviderSet.ANY))
        .isFalse();
    assertThat(noAspects.getRequiredProvidersForAspects()
                        .isSatisfiedBy(AdvertisedProviderSet.EMPTY))
        .isFalse();

    assertThat(noAspects.getRequiredProvidersForAspects().isSatisfiedBy(expectedFailSet))
        .isFalse();
  }

  @Test
  public void testNoConfigurationFragmentPolicySetup_hasNonNullPolicy() throws Exception {
    AspectDefinition noPolicy = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .build();
    assertThat(noPolicy.getConfigurationFragmentPolicy()).isNotNull();
  }

  @Test
  public void testRequiresConfigurationFragments_propagatedToConfigurationFragmentPolicy()
      throws Exception {
    AspectDefinition requiresFragments = new AspectDefinition.Builder(TEST_ASPECT_CLASS)
        .requiresConfigurationFragments(Integer.class, String.class)
        .build();
    assertThat(requiresFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
        requiresFragments.getConfigurationFragmentPolicy().getRequiredConfigurationFragments())
            .containsExactly(Integer.class, String.class);
  }

  private static class FooFragment extends Fragment {}

  private static class BarFragment extends Fragment {}

  @Test
  public void testMissingFragmentPolicy_propagatedToConfigurationFragmentPolicy() throws Exception {
    AspectDefinition missingFragments =
        new AspectDefinition.Builder(TEST_ASPECT_CLASS)
            .setMissingFragmentPolicy(FooFragment.class, MissingFragmentPolicy.IGNORE)
            .build();
    assertThat(missingFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
            missingFragments
                .getConfigurationFragmentPolicy()
                .getMissingFragmentPolicy(FooFragment.class))
        .isEqualTo(MissingFragmentPolicy.IGNORE);
  }

  @Test
  public void testRequiresHostConfigurationFragments_propagatedToConfigurationFragmentPolicy()
      throws Exception {
    AspectDefinition requiresFragments =
        ConfigAwareAspectBuilder.of(new AspectDefinition.Builder(TEST_ASPECT_CLASS))
            .requiresHostConfigurationFragments(FooFragment.class, BarFragment.class)
            .originalBuilder()
            .build();
    assertThat(requiresFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
        requiresFragments.getConfigurationFragmentPolicy().getRequiredConfigurationFragments())
            .containsExactly(FooFragment.class, BarFragment.class);
  }

  @Test
  public void testRequiresConfigurationFragmentNames_propagatedToConfigurationFragmentPolicy()
      throws Exception {
    AspectDefinition requiresFragments =
        new AspectDefinition.Builder(TEST_ASPECT_CLASS)
            .requiresConfigurationFragmentsByStarlarkBuiltinName(ImmutableList.of("test_fragment"))
            .build();
    assertThat(requiresFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
        requiresFragments.getConfigurationFragmentPolicy()
            .isLegalConfigurationFragment(TestFragment.class, NoTransition.INSTANCE))
        .isTrue();
  }

  @Test
  public void testRequiresHostConfigurationFragmentNames_propagatedToConfigurationFragmentPolicy()
      throws Exception {
    AspectDefinition requiresFragments =
        ConfigAwareAspectBuilder.of(new AspectDefinition.Builder(TEST_ASPECT_CLASS))
            .requiresHostConfigurationFragmentsByStarlarkBuiltinName(
                ImmutableList.of("test_fragment"))
            .originalBuilder()
            .build();
    assertThat(requiresFragments.getConfigurationFragmentPolicy()).isNotNull();
    assertThat(
        requiresFragments.getConfigurationFragmentPolicy()
            .isLegalConfigurationFragment(TestFragment.class, HostTransition.INSTANCE))
        .isTrue();
  }

  @Test
  public void testEmptyStarlarkConfigurationFragmentPolicySetup_hasNonNullPolicy()
      throws Exception {
    AspectDefinition noPolicy =
        ConfigAwareAspectBuilder.of(new AspectDefinition.Builder(TEST_ASPECT_CLASS))
            .requiresHostConfigurationFragmentsByStarlarkBuiltinName(ImmutableList.<String>of())
            .originalBuilder()
            .requiresConfigurationFragmentsByStarlarkBuiltinName(ImmutableList.<String>of())
            .build();
    assertThat(noPolicy.getConfigurationFragmentPolicy()).isNotNull();
  }

  @StarlarkBuiltin(name = "test_fragment", doc = "test fragment")
  private static final class TestFragment implements StarlarkValue {}
}
