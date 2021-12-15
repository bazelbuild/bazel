// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FragmentRegistry}. */
@RunWith(JUnit4.class)
public final class FragmentRegistryTest {

  private static final class OptionsA extends FragmentOptions {}

  private static final class OptionsB extends FragmentOptions {}

  private static final class MoreOptions extends FragmentOptions {}

  private static final class EvenMoreOptions extends FragmentOptions {}

  @RequiresOptions(options = OptionsA.class)
  private static final class FragmentA extends Fragment {}

  @RequiresOptions(options = OptionsB.class)
  private static final class FragmentB extends Fragment {}

  @Test
  public void createsRegistry() {
    FragmentRegistry registry =
        FragmentRegistry.create(
            /*allFragments=*/ ImmutableList.of(FragmentA.class, FragmentB.class),
            /*universalFragments=*/ ImmutableList.of(FragmentB.class),
            /*additionalOptions=*/ ImmutableList.of(MoreOptions.class, EvenMoreOptions.class));

    assertThat(registry.getAllFragments()).containsExactly(FragmentA.class, FragmentB.class);
    assertThat(registry.getUniversalFragments()).containsExactly(FragmentB.class);
    assertThat(registry.getOptionsClasses())
        .containsExactly(OptionsA.class, OptionsB.class, MoreOptions.class, EvenMoreOptions.class);
  }

  @Test
  public void canonicalizesOrder() {
    FragmentRegistry registry1 =
        FragmentRegistry.create(
            /*allFragments=*/ ImmutableList.of(FragmentA.class, FragmentB.class),
            /*universalFragments=*/ ImmutableList.of(FragmentA.class, FragmentB.class),
            /*additionalOptions=*/ ImmutableList.of(MoreOptions.class, EvenMoreOptions.class));
    FragmentRegistry registry2 =
        FragmentRegistry.create(
            /*allFragments=*/ ImmutableList.of(FragmentB.class, FragmentA.class),
            /*universalFragments=*/ ImmutableList.of(FragmentB.class, FragmentA.class),
            /*additionalOptions=*/ ImmutableList.of(EvenMoreOptions.class, MoreOptions.class));

    assertThat(registry1.getAllFragments())
        .containsAtLeastElementsIn(registry2.getAllFragments())
        .inOrder();
    assertThat(registry1.getUniversalFragments())
        .containsAtLeastElementsIn(registry2.getUniversalFragments())
        .inOrder();
    assertThat(registry1.getOptionsClasses())
        .containsAtLeastElementsIn(registry2.getOptionsClasses())
        .inOrder();
  }

  @Test
  public void allFragmentsMustContainUniversalFragments() {
    Exception e =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                FragmentRegistry.create(
                    /*allFragments=*/ ImmutableList.of(FragmentA.class),
                    /*universalFragments=*/ ImmutableList.of(FragmentB.class),
                    /*additionalOptions=*/ ImmutableList.of()));

    assertThat(e).hasMessageThat().contains("Missing universally required fragments");
    assertThat(e).hasMessageThat().contains("FragmentB");
  }
}
