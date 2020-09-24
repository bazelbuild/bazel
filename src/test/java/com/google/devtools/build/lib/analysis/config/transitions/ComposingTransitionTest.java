// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config.transitions;

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.java.JavaOptions;
import java.util.Map;
import java.util.stream.IntStream;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ComposingTransition}. */
@RunWith(JUnit4.class)
public class ComposingTransitionTest {
  // Use starlark flags for the test since they are easy to set and check.
  private static final Label FLAG_1 = Label.parseAbsoluteUnchecked("//flag1");
  private static final Label FLAG_2 = Label.parseAbsoluteUnchecked("//flag2");
  private EventHandler eventHandler;

  @Before
  public void init() {
    eventHandler = new StoredEventHandler();
  }

  @Test
  public void compose_patch_patch() throws Exception {
    // Same flag, will overwrite.
    ConfigurationTransition composed =
        ComposingTransition.of(new StubPatch(FLAG_1, "value1"), new StubPatch(FLAG_1, "value2"));

    assertThat(composed).isNotNull();
    Map<String, BuildOptions> results =
        composed.apply(
            TransitionUtil.restrict(composed, BuildOptions.builder().build()), eventHandler);
    assertThat(results).isNotNull();
    assertThat(results).hasSize(1);
    BuildOptions result = Iterables.getOnlyElement(results.values());
    assertThat(result).isNotNull();
    assertThat(result.getStarlarkOptions()).containsEntry(FLAG_1, "value2");
  }

  @Test
  public void compose_patch_split() throws Exception {
    // Different flags, will combine.
    ConfigurationTransition composed =
        ComposingTransition.of(
            new StubPatch(FLAG_1, "value1"), new StubSplit(FLAG_2, "value2a", "value2b"));

    assertThat(composed).isNotNull();
    Map<String, BuildOptions> results =
        composed.apply(
            TransitionUtil.restrict(composed, BuildOptions.builder().build()), eventHandler);
    assertThat(results).isNotNull();
    assertThat(results).hasSize(2);

    BuildOptions result0 = results.get("stub_split0");
    assertThat(result0).isNotNull();
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_1, "value1");
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_2, "value2a");

    BuildOptions result1 = results.get("stub_split1");
    assertThat(result1).isNotNull();
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_1, "value1");
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_2, "value2b");
  }

  @Test
  public void compose_split_patch() throws Exception {
    // Different flags, will combine.
    ConfigurationTransition composed =
        ComposingTransition.of(
            new StubSplit(FLAG_1, "value1a", "value1b"), new StubPatch(FLAG_2, "value2"));

    assertThat(composed).isNotNull();
    Map<String, BuildOptions> results =
        composed.apply(
            TransitionUtil.restrict(composed, BuildOptions.builder().build()), eventHandler);
    assertThat(results).isNotNull();
    assertThat(results).hasSize(2);

    BuildOptions result0 = results.get("stub_split0");
    assertThat(result0).isNotNull();
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_1, "value1a");
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_2, "value2");

    BuildOptions result1 = results.get("stub_split1");
    assertThat(result1).isNotNull();
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_1, "value1b");
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_2, "value2");
  }

  @Test
  public void compose_split_split_disallowed() {
    // Combining two split transitions is not allowed.
    ConfigurationTransition composed =
        ComposingTransition.of(
            new StubSplit(FLAG_1, "value1a", "value1b"),
            new StubSplit(FLAG_2, "value2a", "value2b"));

    assertThat(composed).isNotNull();
    assertThrows(
        IllegalStateException.class,
        () ->
            composed.apply(
                TransitionUtil.restrict(composed, BuildOptions.builder().build()), eventHandler));
  }

  @Test
  public void compose_host_first() {
    ConfigurationTransition composed =
        ComposingTransition.of(HostTransition.INSTANCE, new StubPatch(FLAG_1, "value2"));

    assertThat(composed).isNotNull();
    assertThat(composed.isHostTransition()).isTrue();
  }

  @Test
  public void compose_host_second() {
    ConfigurationTransition composed =
        ComposingTransition.of(new StubPatch(FLAG_1, "value2"), HostTransition.INSTANCE);

    assertThat(composed).isNotNull();
    assertThat(composed.isHostTransition()).isTrue();
  }

  @Test
  public void compose_noTrans_first() {
    StubPatch patch = new StubPatch(FLAG_1, "value");
    ConfigurationTransition composed = ComposingTransition.of(NoTransition.INSTANCE, patch);

    assertThat(composed).isNotNull();
    assertThat(composed).isEqualTo(patch);
  }

  @Test
  public void compose_noTrans_second() {
    StubPatch patch = new StubPatch(FLAG_1, "value");
    ConfigurationTransition composed = ComposingTransition.of(patch, NoTransition.INSTANCE);

    assertThat(composed).isNotNull();
    assertThat(composed).isEqualTo(patch);
  }

  // Helper methods and classes for the tests.
  private static BuildOptions updateOptions(BuildOptions source, Label flag, String value) {
    return source.clone().toBuilder().addStarlarkOption(flag, value).build();
  }

  private static final class StubPatch implements PatchTransition {
    private final Label flagLabel;
    private final String flagValue;

    StubPatch(Label flagLabel, String flagValue) {
      this.flagLabel = flagLabel;
      this.flagValue = flagValue;
    }

    @Override
    public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
      return updateOptions(options.underlying(), flagLabel, flagValue);
    }
  }

  private static final class StubSplit implements SplitTransition {
    private final Label flagLabel;
    private final ImmutableList<String> flagValues;

    StubSplit(Label flagLabel, String... flagValues) {
      this.flagLabel = flagLabel;
      this.flagValues = ImmutableList.copyOf(flagValues);
    }

    @Override
    public ImmutableMap<String, BuildOptions> split(
        BuildOptionsView options, EventHandler eventHandler) {
      return IntStream.range(0, flagValues.size())
          .boxed()
          .collect(
              toImmutableMap(
                  i -> "stub_split" + i,
                  i -> updateOptions(options.underlying(), flagLabel, flagValues.get(i))));
    }
  }

  private static final class TransitionWithCustomFragments implements PatchTransition {
    private final ImmutableSet<Class<? extends FragmentOptions>> fragments;

    TransitionWithCustomFragments(ImmutableSet<Class<? extends FragmentOptions>> fragments) {
      this.fragments = fragments;
    }

    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return fragments;
    }

    @Override
    public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
      return options.underlying();
    }
  }

  @Test
  public void composed_required_fragments() throws Exception {
    ConfigurationTransition composed =
        ComposingTransition.of(
            new TransitionWithCustomFragments(ImmutableSet.of(CppOptions.class)),
            new TransitionWithCustomFragments(ImmutableSet.of(JavaOptions.class)));
    assertThat(composed.requiresOptionFragments(BuildOptions.builder().build()))
        .containsExactly("CppOptions", "JavaOptions");
  }
}
