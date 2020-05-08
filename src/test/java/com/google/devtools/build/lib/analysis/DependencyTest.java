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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.common.testing.NullPointerTester;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
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
public class DependencyTest extends AnalysisTestCase {
  @Test
  public void withNullConfiguration_BasicAccessors() throws Exception {
    Dependency nullDep =
            Dependency.builder(Label.parseAbsolute("//a", ImmutableMap.of()))
                    .withNullConfiguration()
                    .build();

    assertThat(nullDep.getLabel()).isEqualTo(Label.parseAbsolute("//a", ImmutableMap.of()));
    assertThat(nullDep.getConfiguration()).isNull();
    assertThat(nullDep.getAspects().getAllAspects()).isEmpty();
  }

  @Test
  public void withConfiguration_BasicAccessors() throws Exception {
    update();
    Dependency targetDep =
            Dependency.builder(Label.parseAbsolute("//a", ImmutableMap.of()))
                .setConfiguration(getTargetConfiguration())
                    .build();

    assertThat(targetDep.getLabel()).isEqualTo(Label.parseAbsolute("//a", ImmutableMap.of()));
    assertThat(targetDep.getConfiguration()).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspects().getAllAspects()).isEmpty();
  }

  @Test
  public void withConfigurationAndAspects_BasicAccessors() throws Exception {
    update();
    AspectDescriptor simpleAspect = new AspectDescriptor(TestAspects.SIMPLE_ASPECT);
    AspectDescriptor attributeAspect = new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT);
    AspectCollection twoAspects = AspectCollection.createForTests(
        ImmutableSet.of(simpleAspect, attributeAspect));
    Dependency targetDep =
            Dependency.builder(Label.parseAbsolute("//a", ImmutableMap.of()))
                .setConfiguration(getTargetConfiguration())
                    .addAspects(twoAspects)
                    .build();

    assertThat(targetDep.getLabel()).isEqualTo(Label.parseAbsolute("//a", ImmutableMap.of()));
    assertThat(targetDep.getConfiguration()).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspects()).isEqualTo(twoAspects);
    assertThat(targetDep.getAspectConfiguration(simpleAspect)).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspectConfiguration(attributeAspect))
        .isEqualTo(getTargetConfiguration());
  }

  @Test
  public void withConfigurationAndAspects_RejectsNullConfigWithNPE() throws Exception {
    // Although the NullPointerTester should check this, this test invokes a different code path,
    // because it includes aspects (which the NPT test will not).
    AspectDescriptor simpleAspect = new AspectDescriptor(TestAspects.SIMPLE_ASPECT);
    AspectDescriptor attributeAspect = new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT);
    AspectCollection twoAspects = AspectCollection.createForTests(simpleAspect, attributeAspect);

    assertThrows(
        NullPointerException.class,
        () -> Dependency.builder(Label.parseAbsolute("//a", ImmutableMap.of()))
            .setConfiguration(null)
                    .addAspects(twoAspects)
                    .build());
  }

  @Test
  public void withConfigurationAndAspects_AllowsEmptyAspectSet() throws Exception {
    update();
    Dependency dep =
            Dependency.builder(Label.parseAbsolute("//a", ImmutableMap.of()))
                .setConfiguration(getTargetConfiguration())
                    .addAspects(AspectCollection.EMPTY)
                    .build();
    // Here we're also checking that this doesn't throw an exception. No boom? OK. Good.
    assertThat(dep.getAspects().getAllAspects()).isEmpty();
  }

  @Test
  public void withConfiguredAspects_BasicAccessors() throws Exception {
    update();
    AspectDescriptor simpleAspect = new AspectDescriptor(TestAspects.SIMPLE_ASPECT);
    AspectDescriptor attributeAspect = new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT);
    AspectCollection aspects =
        AspectCollection.createForTests(ImmutableSet.of(simpleAspect, attributeAspect));
    ImmutableMap<AspectDescriptor, BuildConfiguration> twoAspectMap = ImmutableMap.of(
        simpleAspect, getTargetConfiguration(), attributeAspect, getHostConfiguration());
    Dependency targetDep =
            Dependency.builder(Label.parseAbsolute("//a", ImmutableMap.of()))
                .setConfiguration(getTargetConfiguration())
                    .addAspects(aspects)
                    .addAspectConfigurations(twoAspectMap)
                    .build();

    assertThat(targetDep.getLabel()).isEqualTo(Label.parseAbsolute("//a", ImmutableMap.of()));
    assertThat(targetDep.getConfiguration()).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspects().getAllAspects())
        .containsExactly(simpleAspect, attributeAspect);
    assertThat(targetDep.getAspectConfiguration(simpleAspect)).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspectConfiguration(attributeAspect))
        .isEqualTo(getHostConfiguration());
  }


  @Test
  public void withConfiguredAspects_AllowsEmptyAspectMap() throws Exception {
    update();
    Dependency dep =
            Dependency.builder(Label.parseAbsolute("//a", ImmutableMap.of()))
                .setConfiguration(getTargetConfiguration())
                    .addAspects(AspectCollection.EMPTY)
                    .addAspectConfigurations(ImmutableMap.<AspectDescriptor, BuildConfiguration>of())
                    .build();
    // Here we're also checking that this doesn't throw an exception. No boom? OK. Good.
    assertThat(dep.getAspects().getAllAspects()).isEmpty();
  }

  @Test
  public void factoriesPassNullableTester() throws Exception {
    update();

    new NullPointerTester()
        .setDefault(Label.class, Label.parseAbsolute("//a", ImmutableMap.of()))
        .setDefault(BuildConfiguration.class, getTargetConfiguration())
        .testAllPublicStaticMethods(Dependency.class);
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
            // base set: //a, host configuration, normal aspect set
                Dependency.builder(a).setConfiguration(host)
                        .addAspects(twoAspects)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(host)
                        .addAspects(twoAspects)
                        .build(),
                Dependency.builder(a).setConfiguration(host)
                        .addAspects(inverseAspects)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(host)
                        .addAspects(inverseAspects)
                        .build(),
                Dependency.builder(a).setConfiguration(host)
                        .addAspects(twoAspects)
                        .addAspectConfigurations(twoAspectsHostMap)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(host)
                        .addAspects(twoAspects)
                        .addAspectConfigurations(twoAspectsHostMap)
                        .build())
        .addEqualityGroup(
            // base set but with label //b
                Dependency.builder(b).setConfiguration(host)
                        .addAspects(twoAspects)
                        .build(),
                Dependency.builder(b).setConfiguration(host)
                        .addAspects(inverseAspects)
                        .build(),
                Dependency.builder(b).setConfiguration(host)
                        .addAspects(twoAspects)
                        .addAspectConfigurations(twoAspectsHostMap)
                        .build())
        .addEqualityGroup(
            // base set but with target configuration
                Dependency.builder(a).setConfiguration(target)
                        .addAspects(twoAspects)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(target)
                        .addAspects(twoAspects)
                        .build(),
                Dependency.builder(a).setConfiguration(target)
                        .addAspects(inverseAspects)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(target)
                        .addAspects(inverseAspects)
                        .build(),
                Dependency.builder(a).setConfiguration(target)
                        .addAspects(twoAspects)
                        .addAspectConfigurations(twoAspectsTargetMap)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(target)
                        .addAspects(twoAspects)
                        .addAspectConfigurations(twoAspectsTargetMap)
                        .build())
        .addEqualityGroup(
            // base set but with null configuration
                Dependency.builder(a)
                        .withNullConfiguration()
                        .build(),
                Dependency.builder(aExplicit)
                        .withNullConfiguration()
                        .build())
        .addEqualityGroup(
            // base set but with different aspects
                Dependency.builder(a).setConfiguration(host)
                        .addAspects(differentAspects)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(host)
                        .addAspects(differentAspects)
                        .build(),
                Dependency.builder(a).setConfiguration(host)
                        .addAspects(differentAspects)
                        .addAspectConfigurations(differentAspectsHostMap)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(host)
                        .addAspects(differentAspects)
                        .addAspectConfigurations(differentAspectsHostMap)
                        .build())
        .addEqualityGroup(
            // base set but with label //b and target configuration
                Dependency.builder(b).setConfiguration(target)
                        .addAspects(twoAspects)
                        .build(),
                Dependency.builder(b).setConfiguration(target)
                        .addAspects(inverseAspects)
                        .build(),
                Dependency.builder(b).setConfiguration(target)
                        .addAspects(twoAspects)
                        .addAspectConfigurations(twoAspectsTargetMap)
                        .build())
        .addEqualityGroup(
            // base set but with label //b and null configuration
                Dependency.builder(b)
                        .withNullConfiguration()
                        .build())
        .addEqualityGroup(
            // base set but with label //b and different aspects
                Dependency.builder(b).setConfiguration(host)
                        .addAspects(differentAspects)
                        .build(),
                Dependency.builder(b).setConfiguration(host)
                        .addAspects(differentAspects)
                        .addAspectConfigurations(differentAspectsHostMap)
                        .build())
        .addEqualityGroup(
            // base set but with target configuration and different aspects
                Dependency.builder(a).setConfiguration(target)
                        .addAspects(differentAspects)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(target)
                        .addAspects(differentAspects)
                        .build(),
                Dependency.builder(a).setConfiguration(target)
                        .addAspects(differentAspects)
                        .addAspectConfigurations(differentAspectsTargetMap)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(target)
                        .addAspects(differentAspects)
                        .addAspectConfigurations(differentAspectsTargetMap)
                        .build())
        .addEqualityGroup(
            // inverse of base set: //b, target configuration, different aspects
                Dependency.builder(b).setConfiguration(target)
                        .addAspects(differentAspects)
                        .build(),
                Dependency.builder(b).setConfiguration(target)
                        .addAspects(differentAspects)
                        .addAspectConfigurations(differentAspectsTargetMap)
                        .build())
        .addEqualityGroup(
            // base set but with no aspects
                Dependency.builder(a).setConfiguration(host)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(host)
                        .build(),
                Dependency.builder(a).setConfiguration(host)
                        .addAspects(noAspects)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(host)
                        .addAspects(noAspects)
                        .build(),
                Dependency.builder(a).setConfiguration(host)
                        .addAspects(noAspects)
                        .addAspectConfigurations(noAspectsMap)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(host)
                        .addAspects(noAspects)
                        .addAspectConfigurations(noAspectsMap)
                        .build())
        .addEqualityGroup(
            // base set but with label //b and no aspects
                Dependency.builder(b).setConfiguration(host)
                        .build(),
                Dependency.builder(b).setConfiguration(host)
                        .addAspects(noAspects)
                        .build(),
                Dependency.builder(b).setConfiguration(host)
                        .addAspects(noAspects)
                        .addAspectConfigurations(noAspectsMap)
                        .build())
        .addEqualityGroup(
            // base set but with target configuration and no aspects
                Dependency.builder(a).setConfiguration(target)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(target)
                        .build(),
                Dependency.builder(a).setConfiguration(target)
                        .addAspects(noAspects)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(target)
                        .addAspects(noAspects)
                        .build(),
                Dependency.builder(a).setConfiguration(target)
                        .addAspects(noAspects)
                        .addAspectConfigurations(noAspectsMap)
                        .build(),
                Dependency.builder(aExplicit).setConfiguration(target)
                        .addAspects(noAspects)
                        .addAspectConfigurations(noAspectsMap)
                        .build())
        .addEqualityGroup(
            // inverse of base set: //b, target configuration, no aspects
                Dependency.builder(b).setConfiguration(target)
                        .build(),
                Dependency.builder(b).setConfiguration(target)
                        .addAspects(noAspects)
                        .build(),
                Dependency.builder(b).setConfiguration(target)
                        .addAspects(noAspects)
                        .addAspectConfigurations(noAspectsMap)
                        .build())
        .testEquals();
  }
}
