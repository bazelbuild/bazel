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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.common.testing.NullPointerTester;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link Dependency}.
 *
 * <p>Although this is just a data class, we need a way to create a configuration.
 */
@RunWith(JUnit4.class)
public class ConfigurationTransitionDependencyTest extends AnalysisTestCase {
  @Test
  public void withTransitionAndAspects_BasicAccessors() throws Exception {
    AspectDescriptor simpleAspect = new AspectDescriptor(TestAspects.SIMPLE_ASPECT);
    AspectDescriptor attributeAspect = new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT);
    AspectCollection twoAspects = AspectCollection.createForTests(
        ImmutableSet.of(simpleAspect, attributeAspect));
    ConfigurationTransitionDependency hostDep =
        ConfigurationTransitionDependency.builder()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
                    .setTransition(HostTransition.INSTANCE)
                    .addAspects(twoAspects)
                    .build();

    assertThat(hostDep.getLabel()).isEqualTo(Label.parseAbsolute("//a", ImmutableMap.of()));
    assertThat(hostDep.hasExplicitConfiguration()).isFalse();
    assertThat(hostDep.getAspects().getAllAspects())
        .containsExactlyElementsIn(twoAspects.getAllAspects());
    assertThat(hostDep.getTransition().isHostTransition()).isTrue();

    assertThrows(IllegalStateException.class, () -> hostDep.getConfiguration());

    assertThrows(IllegalStateException.class, () -> hostDep.getAspectConfiguration(simpleAspect));

    assertThrows(
        IllegalStateException.class, () -> hostDep.getAspectConfiguration(attributeAspect));
  }

  @Test
  public void withTransitionAndAspects_AllowsEmptyAspectSet() throws Exception {
    update();
    ConfigurationTransitionDependency dep =
        ConfigurationTransitionDependency.builder()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
                    .setTransition(HostTransition.INSTANCE)
                    .addAspects(AspectCollection.EMPTY)
                    .build();
    // Here we're also checking that this doesn't throw an exception. No boom? OK. Good.
    assertThat(dep.getAspects().getAllAspects()).isEmpty();
  }

  @Test
  public void factoriesPassNullableTester() throws Exception {
    update();

    new NullPointerTester()
        .setDefault(Label.class, Label.parseAbsolute("//a", ImmutableMap.of()))
        .testAllPublicStaticMethods(ConfigurationTransitionDependency.class);
  }

  @Test
  public void equalsPassesEqualsTester() throws Exception {
    update();

    Label a = Label.parseAbsolute("//a", ImmutableMap.of());
    Label aExplicit = Label.parseAbsolute("//a:a", ImmutableMap.of());
    Label b = Label.parseAbsolute("//b", ImmutableMap.of());

    BuildConfiguration host = getHostConfiguration();
    BuildConfiguration target = getTargetConfiguration();

    AspectDescriptor simpleAspect = new AspectDescriptor(TestAspects.SIMPLE_ASPECT);
    AspectDescriptor attributeAspect = new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT);
    AspectDescriptor errorAspect = new AspectDescriptor(TestAspects.ERROR_ASPECT);

    AspectCollection twoAspects =
        AspectCollection.createForTests(simpleAspect, attributeAspect);
    AspectCollection inverseAspects =
        AspectCollection.createForTests(attributeAspect, simpleAspect);
    AspectCollection differentAspects =
        AspectCollection.createForTests(attributeAspect, errorAspect);
    AspectCollection noAspects = AspectCollection.EMPTY;

    ImmutableMap<AspectDescriptor, BuildConfiguration> twoAspectsHostMap =
        ImmutableMap.of(simpleAspect, host, attributeAspect, host);
    ImmutableMap<AspectDescriptor, BuildConfiguration> twoAspectsTargetMap =
        ImmutableMap.of(simpleAspect, target, attributeAspect, target);
    ImmutableMap<AspectDescriptor, BuildConfiguration> differentAspectsHostMap =
        ImmutableMap.of(attributeAspect, host, errorAspect, host);
    ImmutableMap<AspectDescriptor, BuildConfiguration> differentAspectsTargetMap =
        ImmutableMap.of(attributeAspect, target, errorAspect, target);
    ImmutableMap<AspectDescriptor, BuildConfiguration> noAspectsMap =
        ImmutableMap.<AspectDescriptor, BuildConfiguration>of();

    new EqualsTester()
        .addEqualityGroup(
            // base set but with transition HOST
            ConfigurationTransitionDependency.builder()
                .setLabel(a)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(twoAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(aExplicit)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(twoAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(a)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(inverseAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(aExplicit)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(inverseAspects)
                        .build())
        .addEqualityGroup(
            // base set but with transition HOST and different aspects
            ConfigurationTransitionDependency.builder()
                .setLabel(a)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(differentAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(aExplicit)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(differentAspects)
                        .build())
        .addEqualityGroup(
            // base set but with transition HOST and label //b
            ConfigurationTransitionDependency.builder()
                .setLabel(b)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(twoAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(b)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(inverseAspects)
                        .build())
        .addEqualityGroup(
            // inverse of base set: transition HOST, label //b, different aspects
            ConfigurationTransitionDependency.builder()
                .setLabel(b)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(differentAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(b)
                        .setTransition(HostTransition.INSTANCE)
                        .addAspects(differentAspects)
                        .build())
        .addEqualityGroup(
            // base set but with transition NONE
            ConfigurationTransitionDependency.builder()
                .setLabel(a)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(twoAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(aExplicit)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(twoAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(a)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(inverseAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(aExplicit)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(inverseAspects)
                        .build())
        .addEqualityGroup(
            // base set but with transition NONE and different aspects
            ConfigurationTransitionDependency.builder()
                .setLabel(a)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(differentAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(aExplicit)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(differentAspects)
                        .build())
        .addEqualityGroup(
            // base set but with transition NONE and label //b
            ConfigurationTransitionDependency.builder()
                .setLabel(b)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(twoAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(b)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(inverseAspects)
                        .build())
        .addEqualityGroup(
            // inverse of base set: transition NONE, label //b, different aspects
            ConfigurationTransitionDependency.builder()
                .setLabel(b)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(differentAspects)
                        .build(),
            ConfigurationTransitionDependency.builder()
                .setLabel(b)
                        .setTransition(NoTransition.INSTANCE)
                        .addAspects(differentAspects)
                        .build())
        .testEquals();
  }
}
