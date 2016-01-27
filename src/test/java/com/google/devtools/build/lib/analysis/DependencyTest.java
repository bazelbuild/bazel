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
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.common.testing.NullPointerTester;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.NativeAspectClass;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Tests for {@link Dependency}.
 *
 * <p>Although this is just a data class, we need a way to create a configuration.
 */
@RunWith(JUnit4.class)
public class DependencyTest extends AnalysisTestCase {
  @Test
  public void withNullConfiguration_BasicAccessors() throws Exception {
    Dependency nullDep = Dependency.withNullConfiguration(Label.parseAbsolute("//a"));

    assertThat(nullDep.getLabel()).isEqualTo(Label.parseAbsolute("//a"));
    assertThat(nullDep.hasStaticConfiguration()).isTrue();
    assertThat(nullDep.getConfiguration()).isNull();
    assertThat(nullDep.getAspects()).isEmpty();
    assertThat(nullDep.getAspectConfigurations()).isEmpty();

    try {
      nullDep.getTransition();
      fail("withNullConfiguration-created Dependencies should throw ISE on getTransition()");
    } catch (IllegalStateException ex) {
      // good. expected.
    }
  }

  @Test
  public void withConfiguration_BasicAccessors() throws Exception {
    update();
    Dependency targetDep =
        Dependency.withConfiguration(Label.parseAbsolute("//a"), getTargetConfiguration());

    assertThat(targetDep.getLabel()).isEqualTo(Label.parseAbsolute("//a"));
    assertThat(targetDep.hasStaticConfiguration()).isTrue();
    assertThat(targetDep.getConfiguration()).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspects()).isEmpty();
    assertThat(targetDep.getAspectConfigurations()).isEmpty();

    try {
      targetDep.getTransition();
      fail("withConfiguration-created Dependencies should throw ISE on getTransition()");
    } catch (IllegalStateException ex) {
      // good. expected.
    }
  }

  @Test
  public void withConfigurationAndAspects_BasicAccessors() throws Exception {
    update();
    Aspect simpleAspect = new Aspect(
        new NativeAspectClass<TestAspects.SimpleAspect>(TestAspects.SimpleAspect.class));
    Aspect attributeAspect = new Aspect(
        new NativeAspectClass<TestAspects.AttributeAspect>(TestAspects.AttributeAspect.class));
    ImmutableSet<Aspect> twoAspects = ImmutableSet.of(simpleAspect, attributeAspect);
    Dependency targetDep =
        Dependency.withConfigurationAndAspects(
            Label.parseAbsolute("//a"), getTargetConfiguration(), twoAspects);

    assertThat(targetDep.getLabel()).isEqualTo(Label.parseAbsolute("//a"));
    assertThat(targetDep.hasStaticConfiguration()).isTrue();
    assertThat(targetDep.getConfiguration()).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspects()).containsExactlyElementsIn(twoAspects);
    assertThat(targetDep.getAspectConfigurations())
        .containsExactlyEntriesIn(
            ImmutableMap.of(
                simpleAspect, getTargetConfiguration(),
                attributeAspect, getTargetConfiguration()));

    try {
      targetDep.getTransition();
      fail("withConfigurationAndAspects-created Dependencies should throw ISE on getTransition()");
    } catch (IllegalStateException ex) {
      // good. that's what I WANTED to happen.
    }
  }

  @Test
  public void withConfigurationAndAspects_RejectsNullAspectsWithNPE() throws Exception {
    update();
    Set<Aspect> nullSet = new LinkedHashSet<>();
    nullSet.add(null);

    try {
      Dependency.withConfigurationAndAspects(
          Label.parseAbsolute("//a"), getTargetConfiguration(), nullSet);
      fail("should not be allowed to create a dependency with a null aspect");
    } catch (NullPointerException expected) {
      // good. just as planned.
    }
  }

  @Test
  public void withConfigurationAndAspects_RejectsNullConfigWithNPE() throws Exception {
    // Although the NullPointerTester should check this, this test invokes a different code path,
    // because it includes aspects (which the NPT test will not).
    Aspect simpleAspect = new Aspect(
        new NativeAspectClass<TestAspects.SimpleAspect>(TestAspects.SimpleAspect.class));
    Aspect attributeAspect = new Aspect(
        new NativeAspectClass<TestAspects.AttributeAspect>(TestAspects.AttributeAspect.class));
    ImmutableSet<Aspect> twoAspects = ImmutableSet.of(simpleAspect, attributeAspect);

    try {
      Dependency.withConfigurationAndAspects(Label.parseAbsolute("//a"), null, twoAspects);
      fail("should not be allowed to create a dependency with a null configuration");
    } catch (NullPointerException expected) {
      // good. you fell rrrrright into my trap.
    }
  }

  @Test
  public void withConfigurationAndAspects_AllowsEmptyAspectSet() throws Exception {
    update();
    Dependency dep =
        Dependency.withConfigurationAndAspects(
            Label.parseAbsolute("//a"), getTargetConfiguration(), ImmutableSet.<Aspect>of());
    // Here we're also checking that this doesn't throw an exception. No boom? OK. Good.
    assertThat(dep.getAspects()).isEmpty();
    assertThat(dep.getAspectConfigurations()).isEmpty();
  }

  @Test
  public void withConfiguredAspects_BasicAccessors() throws Exception {
    update();
    Aspect simpleAspect = new Aspect(
        new NativeAspectClass<TestAspects.SimpleAspect>(TestAspects.SimpleAspect.class));
    Aspect attributeAspect = new Aspect(
        new NativeAspectClass<TestAspects.AttributeAspect>(TestAspects.AttributeAspect.class));
    ImmutableMap<Aspect, BuildConfiguration> twoAspectMap = ImmutableMap.of(
        simpleAspect, getTargetConfiguration(), attributeAspect, getHostConfiguration());
    Dependency targetDep =
        Dependency.withConfiguredAspects(
            Label.parseAbsolute("//a"), getTargetConfiguration(), twoAspectMap);

    assertThat(targetDep.getLabel()).isEqualTo(Label.parseAbsolute("//a"));
    assertThat(targetDep.hasStaticConfiguration()).isTrue();
    assertThat(targetDep.getConfiguration()).isEqualTo(getTargetConfiguration());
    assertThat(targetDep.getAspects())
        .containsExactlyElementsIn(ImmutableSet.of(simpleAspect, attributeAspect));
    assertThat(targetDep.getAspectConfigurations()).containsExactlyEntriesIn(twoAspectMap);

    try {
      targetDep.getTransition();
      fail("withConfiguredAspects-created Dependencies should throw ISE on getTransition()");
    } catch (IllegalStateException ex) {
      // good. all according to keikaku. (TL note: keikaku means plan)
    }
  }


  @Test
  public void withConfiguredAspects_AllowsEmptyAspectMap() throws Exception {
    update();
    Dependency dep =
        Dependency.withConfiguredAspects(
            Label.parseAbsolute("//a"), getTargetConfiguration(),
            ImmutableMap.<Aspect, BuildConfiguration>of());
    // Here we're also checking that this doesn't throw an exception. No boom? OK. Good.
    assertThat(dep.getAspects()).isEmpty();
    assertThat(dep.getAspectConfigurations()).isEmpty();
  }

  @Test
  public void withTransitionAndAspects_BasicAccessors() throws Exception {
    Aspect simpleAspect = new Aspect(
        new NativeAspectClass<TestAspects.SimpleAspect>(TestAspects.SimpleAspect.class));
    Aspect attributeAspect = new Aspect(
        new NativeAspectClass<TestAspects.AttributeAspect>(TestAspects.AttributeAspect.class));
    ImmutableSet<Aspect> twoAspects = ImmutableSet.of(simpleAspect, attributeAspect);
    Dependency hostDep =
        Dependency.withTransitionAndAspects(
            Label.parseAbsolute("//a"), ConfigurationTransition.HOST, twoAspects);

    assertThat(hostDep.getLabel()).isEqualTo(Label.parseAbsolute("//a"));
    assertThat(hostDep.hasStaticConfiguration()).isFalse();
    assertThat(hostDep.getAspects()).containsExactlyElementsIn(twoAspects);
    assertThat(hostDep.getTransition()).isEqualTo(ConfigurationTransition.HOST);

    try {
      hostDep.getConfiguration();
      fail("withTransitionAndAspects-created Dependencies should throw ISE on getConfiguration()");
    } catch (IllegalStateException ex) {
      // good. I knew you would do that.
    }

    try {
      hostDep.getAspectConfigurations();
      fail("withTransitionAndAspects-created Dependencies should throw ISE on "
          + "getAspectConfigurations()");
    } catch (IllegalStateException ex) {
      // good. you're so predictable.
    }
  }

  @Test
  public void withTransitionAndAspects_AllowsEmptyAspectSet() throws Exception {
    update();
    Dependency dep =
        Dependency.withTransitionAndAspects(
            Label.parseAbsolute("//a"), ConfigurationTransition.HOST, ImmutableSet.<Aspect>of());
    // Here we're also checking that this doesn't throw an exception. No boom? OK. Good.
    assertThat(dep.getAspects()).isEmpty();
  }

  @Test
  public void factoriesPassNullableTester() throws Exception {
    update();

    new NullPointerTester()
        .setDefault(Label.class, Label.parseAbsolute("//a"))
        .setDefault(BuildConfiguration.class, getTargetConfiguration())
        .testAllPublicStaticMethods(Dependency.class);
  }

  @Test
  public void equalsPassesEqualsTester() throws Exception {
    update();

    Label a = Label.parseAbsolute("//a");
    Label aExplicit = Label.parseAbsolute("//a:a");
    Label b = Label.parseAbsolute("//b");

    BuildConfiguration host = getHostConfiguration();
    BuildConfiguration target = getTargetConfiguration();

    Aspect simpleAspect = new Aspect(
        new NativeAspectClass<TestAspects.SimpleAspect>(TestAspects.SimpleAspect.class));
    Aspect attributeAspect = new Aspect(
        new NativeAspectClass<TestAspects.AttributeAspect>(TestAspects.AttributeAspect.class));
    Aspect errorAspect = new Aspect(
        new NativeAspectClass<TestAspects.ErrorAspect>(TestAspects.ErrorAspect.class));

    ImmutableSet<Aspect> twoAspects = ImmutableSet.of(simpleAspect, attributeAspect);
    ImmutableSet<Aspect> inverseAspects = ImmutableSet.of(attributeAspect, simpleAspect);
    ImmutableSet<Aspect> differentAspects = ImmutableSet.of(attributeAspect, errorAspect);
    ImmutableSet<Aspect> noAspects = ImmutableSet.<Aspect>of();

    ImmutableMap<Aspect, BuildConfiguration> twoAspectsHostMap =
        ImmutableMap.of(simpleAspect, host, attributeAspect, host);
    ImmutableMap<Aspect, BuildConfiguration> twoAspectsTargetMap =
        ImmutableMap.of(simpleAspect, target, attributeAspect, target);
    ImmutableMap<Aspect, BuildConfiguration> differentAspectsHostMap =
        ImmutableMap.of(attributeAspect, host, errorAspect, host);
    ImmutableMap<Aspect, BuildConfiguration> differentAspectsTargetMap =
        ImmutableMap.of(attributeAspect, target, errorAspect, target);
    ImmutableMap<Aspect, BuildConfiguration> noAspectsMap =
        ImmutableMap.<Aspect, BuildConfiguration>of();

    new EqualsTester()
        .addEqualityGroup(
            // base set: //a, host configuration, normal aspect set
            Dependency.withConfigurationAndAspects(a, host, twoAspects),
            Dependency.withConfigurationAndAspects(aExplicit, host, twoAspects),
            Dependency.withConfigurationAndAspects(a, host, inverseAspects),
            Dependency.withConfigurationAndAspects(aExplicit, host, inverseAspects),
            Dependency.withConfiguredAspects(a, host, twoAspectsHostMap),
            Dependency.withConfiguredAspects(aExplicit, host, twoAspectsHostMap))
        .addEqualityGroup(
            // base set but with label //b
            Dependency.withConfigurationAndAspects(b, host, twoAspects),
            Dependency.withConfigurationAndAspects(b, host, inverseAspects),
            Dependency.withConfiguredAspects(b, host, twoAspectsHostMap))
        .addEqualityGroup(
            // base set but with target configuration
            Dependency.withConfigurationAndAspects(a, target, twoAspects),
            Dependency.withConfigurationAndAspects(aExplicit, target, twoAspects),
            Dependency.withConfigurationAndAspects(a, target, inverseAspects),
            Dependency.withConfigurationAndAspects(aExplicit, target, inverseAspects),
            Dependency.withConfiguredAspects(a, target, twoAspectsTargetMap),
            Dependency.withConfiguredAspects(aExplicit, target, twoAspectsTargetMap))
        .addEqualityGroup(
            // base set but with null configuration
            Dependency.withNullConfiguration(a),
            Dependency.withNullConfiguration(aExplicit))
        .addEqualityGroup(
            // base set but with different aspects
            Dependency.withConfigurationAndAspects(a, host, differentAspects),
            Dependency.withConfigurationAndAspects(aExplicit, host, differentAspects),
            Dependency.withConfiguredAspects(a, host, differentAspectsHostMap),
            Dependency.withConfiguredAspects(aExplicit, host, differentAspectsHostMap))
        .addEqualityGroup(
            // base set but with label //b and target configuration
            Dependency.withConfigurationAndAspects(b, target, twoAspects),
            Dependency.withConfigurationAndAspects(b, target, inverseAspects),
            Dependency.withConfiguredAspects(b, target, twoAspectsTargetMap))
        .addEqualityGroup(
            // base set but with label //b and null configuration
            Dependency.withNullConfiguration(b))
        .addEqualityGroup(
            // base set but with label //b and different aspects
            Dependency.withConfigurationAndAspects(b, host, differentAspects),
            Dependency.withConfiguredAspects(b, host, differentAspectsHostMap))
        .addEqualityGroup(
            // base set but with target configuration and different aspects
            Dependency.withConfigurationAndAspects(a, target, differentAspects),
            Dependency.withConfigurationAndAspects(aExplicit, target, differentAspects),
            Dependency.withConfiguredAspects(a, target, differentAspectsTargetMap),
            Dependency.withConfiguredAspects(aExplicit, target, differentAspectsTargetMap))
        .addEqualityGroup(
            // inverse of base set: //b, target configuration, different aspects
            Dependency.withConfigurationAndAspects(b, target, differentAspects),
            Dependency.withConfiguredAspects(b, target, differentAspectsTargetMap))
        .addEqualityGroup(
            // base set but with no aspects
            Dependency.withConfiguration(a, host),
            Dependency.withConfiguration(aExplicit, host),
            Dependency.withConfigurationAndAspects(a, host, noAspects),
            Dependency.withConfigurationAndAspects(aExplicit, host, noAspects),
            Dependency.withConfiguredAspects(a, host, noAspectsMap),
            Dependency.withConfiguredAspects(aExplicit, host, noAspectsMap))
        .addEqualityGroup(
            // base set but with label //b and no aspects
            Dependency.withConfiguration(b, host),
            Dependency.withConfigurationAndAspects(b, host, noAspects),
            Dependency.withConfiguredAspects(b, host, noAspectsMap))
        .addEqualityGroup(
            // base set but with target configuration and no aspects
            Dependency.withConfiguration(a, target),
            Dependency.withConfiguration(aExplicit, target),
            Dependency.withConfigurationAndAspects(a, target, noAspects),
            Dependency.withConfigurationAndAspects(aExplicit, target, noAspects),
            Dependency.withConfiguredAspects(a, target, noAspectsMap),
            Dependency.withConfiguredAspects(aExplicit, target, noAspectsMap))
        .addEqualityGroup(
            // inverse of base set: //b, target configuration, no aspects
            Dependency.withConfiguration(b, target),
            Dependency.withConfigurationAndAspects(b, target, noAspects),
            Dependency.withConfiguredAspects(b, target, noAspectsMap))
        .addEqualityGroup(
            // base set but with transition HOST
            Dependency.withTransitionAndAspects(a, ConfigurationTransition.HOST, twoAspects),
            Dependency.withTransitionAndAspects(
                aExplicit, ConfigurationTransition.HOST, twoAspects),
            Dependency.withTransitionAndAspects(a, ConfigurationTransition.HOST, inverseAspects),
            Dependency.withTransitionAndAspects(
                aExplicit, ConfigurationTransition.HOST, inverseAspects))
        .addEqualityGroup(
            // base set but with transition HOST and different aspects
            Dependency.withTransitionAndAspects(a, ConfigurationTransition.HOST, differentAspects),
            Dependency.withTransitionAndAspects(
                aExplicit, ConfigurationTransition.HOST, differentAspects))
        .addEqualityGroup(
            // base set but with transition HOST and label //b
            Dependency.withTransitionAndAspects(b, ConfigurationTransition.HOST, twoAspects),
            Dependency.withTransitionAndAspects(b, ConfigurationTransition.HOST, inverseAspects))
        .addEqualityGroup(
            // inverse of base set: transition HOST, label //b, different aspects
            Dependency.withTransitionAndAspects(b, ConfigurationTransition.HOST, differentAspects),
            Dependency.withTransitionAndAspects(b, ConfigurationTransition.HOST, differentAspects))
        .addEqualityGroup(
            // base set but with transition NONE
            Dependency.withTransitionAndAspects(a, ConfigurationTransition.NONE, twoAspects),
            Dependency.withTransitionAndAspects(
                aExplicit, ConfigurationTransition.NONE, twoAspects),
            Dependency.withTransitionAndAspects(a, ConfigurationTransition.NONE, inverseAspects),
            Dependency.withTransitionAndAspects(
                aExplicit, ConfigurationTransition.NONE, inverseAspects))
        .addEqualityGroup(
            // base set but with transition NONE and different aspects
            Dependency.withTransitionAndAspects(a, ConfigurationTransition.NONE, differentAspects),
            Dependency.withTransitionAndAspects(
                aExplicit, ConfigurationTransition.NONE, differentAspects))
        .addEqualityGroup(
            // base set but with transition NONE and label //b
            Dependency.withTransitionAndAspects(b, ConfigurationTransition.NONE, twoAspects),
            Dependency.withTransitionAndAspects(b, ConfigurationTransition.NONE, inverseAspects))
        .addEqualityGroup(
            // inverse of base set: transition NONE, label //b, different aspects
            Dependency.withTransitionAndAspects(b, ConfigurationTransition.NONE, differentAspects),
            Dependency.withTransitionAndAspects(b, ConfigurationTransition.NONE, differentAspects))
        .testEquals();
  }
}
