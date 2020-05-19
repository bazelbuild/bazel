// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DependencyKey}. */
@RunWith(JUnit4.class)
public class DependencyKeyTest extends AnalysisTestCase {

  @Test
  public void withTransitionAndAspects_BasicAccessors() throws Exception {
    AspectDescriptor simpleAspect = new AspectDescriptor(TestAspects.SIMPLE_ASPECT);
    AspectDescriptor attributeAspect = new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT);
    AspectCollection twoAspects =
        AspectCollection.createForTests(ImmutableSet.of(simpleAspect, attributeAspect));
    DependencyKey hostDep =
        DependencyKey.builder()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
            .setTransition(HostTransition.INSTANCE)
            .setAspects(twoAspects)
            .build();

    assertThat(hostDep.getLabel()).isEqualTo(Label.parseAbsolute("//a", ImmutableMap.of()));
    assertThat(hostDep.getAspects().getAllAspects())
        .containsExactlyElementsIn(twoAspects.getAllAspects());
    assertThat(hostDep.getTransition().isHostTransition()).isTrue();
  }

  @Test
  public void withTransitionAndAspects_AllowsEmptyAspectSet() throws Exception {
    update();
    DependencyKey dep =
        DependencyKey.builder()
            .setLabel(Label.parseAbsolute("//a", ImmutableMap.of()))
            .setTransition(HostTransition.INSTANCE)
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

    AspectDescriptor simpleAspect = new AspectDescriptor(TestAspects.SIMPLE_ASPECT);
    AspectDescriptor attributeAspect = new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT);
    AspectDescriptor errorAspect = new AspectDescriptor(TestAspects.ERROR_ASPECT);

    AspectCollection twoAspects = AspectCollection.createForTests(simpleAspect, attributeAspect);
    AspectCollection inverseAspects =
        AspectCollection.createForTests(attributeAspect, simpleAspect);
    AspectCollection differentAspects =
        AspectCollection.createForTests(attributeAspect, errorAspect);

    new EqualsTester()
        .addEqualityGroup(
            // base set but with transition HOST
            DependencyKey.builder()
                .setLabel(a)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(twoAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(aExplicit)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(twoAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(a)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(inverseAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(aExplicit)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(inverseAspects)
                .build())
        .addEqualityGroup(
            // base set but with transition HOST and different aspects
            DependencyKey.builder()
                .setLabel(a)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(differentAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(aExplicit)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(differentAspects)
                .build())
        .addEqualityGroup(
            // base set but with transition HOST and label //b
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(twoAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(inverseAspects)
                .build())
        .addEqualityGroup(
            // inverse of base set: transition HOST, label //b, different aspects
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(differentAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(HostTransition.INSTANCE)
                .setAspects(differentAspects)
                .build())
        .addEqualityGroup(
            // base set but with transition NONE
            DependencyKey.builder()
                .setLabel(a)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(twoAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(aExplicit)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(twoAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(a)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(inverseAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(aExplicit)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(inverseAspects)
                .build())
        .addEqualityGroup(
            // base set but with transition NONE and different aspects
            DependencyKey.builder()
                .setLabel(a)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(differentAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(aExplicit)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(differentAspects)
                .build())
        .addEqualityGroup(
            // base set but with transition NONE and label //b
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(twoAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(inverseAspects)
                .build())
        .addEqualityGroup(
            // inverse of base set: transition NONE, label //b, different aspects
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(differentAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(NoTransition.INSTANCE)
                .setAspects(differentAspects)
                .build())
        .testEquals();
  }
}
