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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.AspectCollection.AspectDeps;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DependencyKey}. */
@RunWith(JUnit4.class)
public class DependencyKeyTest extends AnalysisTestCase {
  private static final PatchTransition TEST_TRANSITION =
      (options, eventHandler) -> {
        BuildOptions newOptions = options.underlying().clone();
        newOptions.get(CoreOptions.class).commandLineBuildVariables =
            ImmutableList.of(Map.entry("newkey", "newvalue"));
        return newOptions;
      };

  @Test
  public void withTransitionAndAspects_BasicAccessors() throws Exception {
    AspectDescriptor simpleAspect = new AspectDescriptor(TestAspects.SIMPLE_ASPECT);
    AspectDescriptor attributeAspect = new AspectDescriptor(TestAspects.ATTRIBUTE_ASPECT);
    AspectCollection twoAspects =
        AspectCollection.createForTests(ImmutableSet.of(simpleAspect, attributeAspect));
    DependencyKey transitionDep =
        DependencyKey.builder()
            .setLabel(Label.parseCanonical("//a"))
            .setTransition(TEST_TRANSITION)
            .setAspects(twoAspects)
            .build();

    assertThat(transitionDep.getLabel()).isEqualTo(Label.parseCanonical("//a"));
    assertThat(
            Iterables.transform(transitionDep.getAspects().getUsedAspects(), AspectDeps::getAspect))
        .containsExactlyElementsIn(
            Iterables.transform(twoAspects.getUsedAspects(), AspectDeps::getAspect));
    assertThat(transitionDep.getTransition()).isSameInstanceAs(TEST_TRANSITION);
  }

  @Test
  public void withTransitionAndAspects_AllowsEmptyAspectSet() throws Exception {
    update();
    DependencyKey dep =
        DependencyKey.builder()
            .setLabel(Label.parseCanonical("//a"))
            .setTransition(TEST_TRANSITION)
            .setAspects(AspectCollection.EMPTY)
            .build();
    // Here we're also checking that this doesn't throw an exception. No boom? OK. Good.
    assertThat(dep.getAspects().getUsedAspects()).isEmpty();
  }

  @Test
  public void equalsPassesEqualsTester() throws Exception {
    update();

    Label a = Label.parseCanonical("//a");
    Label aExplicit = Label.parseCanonical("//a:a");
    Label b = Label.parseCanonical("//b");

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
            // base set but with transition
            DependencyKey.builder()
                .setLabel(a)
                .setTransition(TEST_TRANSITION)
                .setAspects(twoAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(aExplicit)
                .setTransition(TEST_TRANSITION)
                .setAspects(twoAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(a)
                .setTransition(TEST_TRANSITION)
                .setAspects(inverseAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(aExplicit)
                .setTransition(TEST_TRANSITION)
                .setAspects(inverseAspects)
                .build())
        .addEqualityGroup(
            // base set but with transition and different aspects
            DependencyKey.builder()
                .setLabel(a)
                .setTransition(TEST_TRANSITION)
                .setAspects(differentAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(aExplicit)
                .setTransition(TEST_TRANSITION)
                .setAspects(differentAspects)
                .build())
        .addEqualityGroup(
            // base set but with transition and label //b
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(TEST_TRANSITION)
                .setAspects(twoAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(TEST_TRANSITION)
                .setAspects(inverseAspects)
                .build())
        .addEqualityGroup(
            // inverse of base set: transition, label //b, different aspects
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(TEST_TRANSITION)
                .setAspects(differentAspects)
                .build(),
            DependencyKey.builder()
                .setLabel(b)
                .setTransition(TEST_TRANSITION)
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
