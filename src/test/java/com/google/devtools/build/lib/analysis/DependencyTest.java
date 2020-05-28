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
        Dependency.builder()
            .withNullConfiguration()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
            .build();

    assertThat(nullDep.getLabel()).isEqualTo(Label.parseAbsolute("//a", ImmutableMap.of()));
    assertThat(nullDep.getConfiguration()).isNull();
    assertThat(nullDep.getAspects().getAllAspects()).isEmpty();
  }

  @Test
  public void withConfiguration_BasicAccessors() throws Exception {
    update();
    Dependency targetDep =
        Dependency.builder()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
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
        Dependency.builder()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
            .setConfiguration(getTargetConfiguration())
            .setAspects(twoAspects)
            .build();

    assertThat(targetDep.getLabel()).isEqualTo(Label.parseAbsolute("//a", ImmutableMap.of()));
    assertThat(targetDep.getConfiguration()).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspects()).isEqualTo(twoAspects);
    assertThat(targetDep.getAspectConfiguration(simpleAspect)).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspectConfiguration(attributeAspect))
        .isEqualTo(getTargetConfiguration());
  }

  @Test
  public void withConfigurationAndAspects_RejectsNullConfig() throws Exception {
    // Although the NullPointerTester should check this, this test invokes a different code path,
    // because it includes aspects (which the NPT test will not).
    AspectDescriptor simpleAspect = new AspectDescriptor(TestAspects.SIMPLE_ASPECT);
    AspectDescriptor attributeAspect = new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT);
    AspectCollection twoAspects = AspectCollection.createForTests(simpleAspect, attributeAspect);

    assertThrows(
        IllegalStateException.class,
        () ->
            Dependency.builder()
                .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
                .setConfiguration(null)
                .setAspects(twoAspects)
                .build());
  }

  @Test
  public void withConfigurationAndAspects_AllowsEmptyAspectSet() throws Exception {
    update();
    Dependency dep =
        Dependency.builder()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
            .setConfiguration(getTargetConfiguration())
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
    Dependency targetDep =
        Dependency.builder()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
            .setConfiguration(getTargetConfiguration())
            .setAspects(aspects)
            .build();

    assertThat(targetDep.getLabel()).isEqualTo(Label.parseAbsolute("//a", ImmutableMap.of()));
    assertThat(targetDep.getConfiguration()).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspects().getAllAspects())
        .containsExactly(simpleAspect, attributeAspect);
  }

  @Test
  public void withConfiguredAspects_AllowsEmptyAspectMap() throws Exception {
    update();
    Dependency dep =
        Dependency.builder()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
            .setConfiguration(getTargetConfiguration())
            .setAspects(AspectCollection.EMPTY)
            .build();
    // Here we're also checking that this doesn't throw an exception. No boom? OK. Good.
    assertThat(dep.getAspects().getAllAspects()).isEmpty();
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

    new EqualsTester()
        .addEqualityGroup(
            // base set: //a, host configuration, normal aspect set
            Dependency.builder().setLabel(a).setConfiguration(host).setAspects(twoAspects).build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(host)
                .setAspects(twoAspects)
                .build(),
            Dependency.builder()
                .setLabel(a)
                .setConfiguration(host)
                .setAspects(inverseAspects)
                .build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(host)
                .setAspects(inverseAspects)
                .build(),
            Dependency.builder().setLabel(a).setConfiguration(host).setAspects(twoAspects).build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(host)
                .setAspects(twoAspects)
                .build())
        .addEqualityGroup(
            // base set but with label //b
            Dependency.builder().setLabel(b).setConfiguration(host).setAspects(twoAspects).build(),
            Dependency.builder()
                .setLabel(b)
                .setConfiguration(host)
                .setAspects(inverseAspects)
                .build(),
            Dependency.builder().setLabel(b).setConfiguration(host).setAspects(twoAspects).build())
        .addEqualityGroup(
            // base set but with target configuration
            Dependency.builder()
                .setLabel(a)
                .setConfiguration(target)
                .setAspects(twoAspects)
                .build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(target)
                .setAspects(twoAspects)
                .build(),
            Dependency.builder()
                .setLabel(a)
                .setConfiguration(target)
                .setAspects(inverseAspects)
                .build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(target)
                .setAspects(inverseAspects)
                .build(),
            Dependency.builder()
                .setLabel(a)
                .setConfiguration(target)
                .setAspects(twoAspects)
                .build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(target)
                .setAspects(twoAspects)
                .build())
        .addEqualityGroup(
            // base set but with null configuration
            Dependency.builder().withNullConfiguration().setLabel(a).build(),
            Dependency.builder().withNullConfiguration().setLabel(aExplicit).build())
        .addEqualityGroup(
            // base set but with different aspects
            Dependency.builder()
                .setLabel(a)
                .setConfiguration(host)
                .setAspects(differentAspects)
                .build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(host)
                .setAspects(differentAspects)
                .build(),
            Dependency.builder()
                .setLabel(a)
                .setConfiguration(host)
                .setAspects(differentAspects)
                .build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(host)
                .setAspects(differentAspects)
                .build())
        .addEqualityGroup(
            // base set but with label //b and target configuration
            Dependency.builder()
                .setLabel(b)
                .setConfiguration(target)
                .setAspects(twoAspects)
                .build(),
            Dependency.builder()
                .setLabel(b)
                .setConfiguration(target)
                .setAspects(inverseAspects)
                .build(),
            Dependency.builder()
                .setLabel(b)
                .setConfiguration(target)
                .setAspects(twoAspects)
                .build())
        .addEqualityGroup(
            // base set but with label //b and null configuration
            Dependency.builder().withNullConfiguration().setLabel(b).build())
        .addEqualityGroup(
            // base set but with label //b and different aspects
            Dependency.builder()
                .setLabel(b)
                .setConfiguration(host)
                .setAspects(differentAspects)
                .build(),
            Dependency.builder()
                .setLabel(b)
                .setConfiguration(host)
                .setAspects(differentAspects)
                .build())
        .addEqualityGroup(
            // base set but with target configuration and different aspects
            Dependency.builder()
                .setLabel(a)
                .setConfiguration(target)
                .setAspects(differentAspects)
                .build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(target)
                .setAspects(differentAspects)
                .build(),
            Dependency.builder()
                .setLabel(a)
                .setConfiguration(target)
                .setAspects(differentAspects)
                .build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(target)
                .setAspects(differentAspects)
                .build())
        .addEqualityGroup(
            // inverse of base set: //b, target configuration, different aspects
            Dependency.builder()
                .setLabel(b)
                .setConfiguration(target)
                .setAspects(differentAspects)
                .build(),
            Dependency.builder()
                .setLabel(b)
                .setConfiguration(target)
                .setAspects(differentAspects)
                .build())
        .addEqualityGroup(
            // base set but with no aspects
            Dependency.builder().setLabel(a).setConfiguration(host).build(),
            Dependency.builder().setLabel(aExplicit).setConfiguration(host).build(),
            Dependency.builder().setLabel(a).setConfiguration(host).setAspects(noAspects).build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(host)
                .setAspects(noAspects)
                .build(),
            Dependency.builder().setLabel(a).setConfiguration(host).setAspects(noAspects).build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(host)
                .setAspects(noAspects)
                .build())
        .addEqualityGroup(
            // base set but with label //b and no aspects
            Dependency.builder().setLabel(b).setConfiguration(host).build(),
            Dependency.builder().setLabel(b).setConfiguration(host).setAspects(noAspects).build(),
            Dependency.builder().setLabel(b).setConfiguration(host).setAspects(noAspects).build())
        .addEqualityGroup(
            // base set but with target configuration and no aspects
            Dependency.builder().setLabel(a).setConfiguration(target).build(),
            Dependency.builder().setLabel(aExplicit).setConfiguration(target).build(),
            Dependency.builder().setLabel(a).setConfiguration(target).setAspects(noAspects).build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(target)
                .setAspects(noAspects)
                .build(),
            Dependency.builder().setLabel(a).setConfiguration(target).setAspects(noAspects).build(),
            Dependency.builder()
                .setLabel(aExplicit)
                .setConfiguration(target)
                .setAspects(noAspects)
                .build())
        .addEqualityGroup(
            // inverse of base set: //b, target configuration, no aspects
            Dependency.builder().setLabel(b).setConfiguration(target).build(),
            Dependency.builder().setLabel(b).setConfiguration(target).setAspects(noAspects).build(),
            Dependency.builder().setLabel(b).setConfiguration(target).setAspects(noAspects).build())
        .testEquals();
  }
}
