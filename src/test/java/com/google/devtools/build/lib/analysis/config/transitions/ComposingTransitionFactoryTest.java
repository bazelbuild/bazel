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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.TransitionFactories;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ComposingTransitionFactory}. */
@RunWith(JUnit4.class)
public class ComposingTransitionFactoryTest {
  // Use starlark flags for the test since they are easy to set and check.
  private static final Label FLAG_1 = Label.parseAbsoluteUnchecked("//flag1");
  private static final Label FLAG_2 = Label.parseAbsoluteUnchecked("//flag2");

  @Test
  public void compose_patch_patch() {
    // Same flag, will overwrite.
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(
            TransitionFactories.of(new StubPatch(FLAG_1, "value1")),
            TransitionFactories.of(new StubPatch(FLAG_1, "value2")));

    assertThat(composed).isNotNull();
    assertThat(composed.isSplit()).isFalse();
    ConfigurationTransition transition = composed.create(new StubData());
    List<BuildOptions> results = transition.apply(BuildOptions.builder().build());
    assertThat(results).isNotNull();
    assertThat(results).hasSize(1);
    BuildOptions result = Iterables.getOnlyElement(results);
    assertThat(result).isNotNull();
    assertThat(result.getStarlarkOptions()).containsEntry(FLAG_1, "value2");
  }

  @Test
  public void compose_patch_split() {
    // Different flags, will combine.
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(
            TransitionFactories.of(new StubPatch(FLAG_1, "value1")),
            TransitionFactories.of(new StubSplit(FLAG_2, "value2a", "value2b")));

    assertThat(composed).isNotNull();
    assertThat(composed.isSplit()).isTrue();
    ConfigurationTransition transition = composed.create(new StubData());
    List<BuildOptions> results = transition.apply(BuildOptions.builder().build());
    assertThat(results).isNotNull();
    assertThat(results).hasSize(2);

    BuildOptions result0 = results.get(0);
    assertThat(result0).isNotNull();
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_1, "value1");
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_2, "value2a");

    BuildOptions result1 = results.get(1);
    assertThat(result1).isNotNull();
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_1, "value1");
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_2, "value2b");
  }

  @Test
  public void compose_split_patch() {
    // Different flags, will combine.
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(
            TransitionFactories.of(new StubSplit(FLAG_1, "value1a", "value1b")),
            TransitionFactories.of(new StubPatch(FLAG_2, "value2")));

    assertThat(composed).isNotNull();
    assertThat(composed.isSplit()).isTrue();
    ConfigurationTransition transition = composed.create(new StubData());
    List<BuildOptions> results = transition.apply(BuildOptions.builder().build());
    assertThat(results).isNotNull();
    assertThat(results).hasSize(2);

    BuildOptions result0 = results.get(0);
    assertThat(result0).isNotNull();
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_1, "value1a");
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_2, "value2");

    BuildOptions result1 = results.get(1);
    assertThat(result1).isNotNull();
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_1, "value1b");
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_2, "value2");
  }

  @Test
  public void compose_split_split() {
    // Different flags, will combine.
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(
            TransitionFactories.of(new StubSplit(FLAG_1, "value1a", "value1b")),
            TransitionFactories.of(new StubSplit(FLAG_2, "value2a", "value2b")));

    assertThat(composed).isNotNull();
    assertThat(composed.isSplit()).isTrue();
    ConfigurationTransition transition = composed.create(new StubData());
    List<BuildOptions> results = transition.apply(BuildOptions.builder().build());
    assertThat(results).isNotNull();
    assertThat(results).hasSize(4);

    BuildOptions result0 = results.get(0);
    assertThat(result0).isNotNull();
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_1, "value1a");
    assertThat(result0.getStarlarkOptions()).containsEntry(FLAG_2, "value2a");

    BuildOptions result1 = results.get(1);
    assertThat(result1).isNotNull();
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_1, "value1a");
    assertThat(result1.getStarlarkOptions()).containsEntry(FLAG_2, "value2b");

    BuildOptions result2 = results.get(2);
    assertThat(result2).isNotNull();
    assertThat(result2.getStarlarkOptions()).containsEntry(FLAG_1, "value1b");
    assertThat(result2.getStarlarkOptions()).containsEntry(FLAG_2, "value2a");

    BuildOptions result3 = results.get(3);
    assertThat(result3).isNotNull();
    assertThat(result3.getStarlarkOptions()).containsEntry(FLAG_1, "value1b");
    assertThat(result3.getStarlarkOptions()).containsEntry(FLAG_2, "value2b");
  }

  @Test
  public void compose_host_first() {
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(
            HostTransition.createFactory(),
            TransitionFactories.of(new StubPatch(FLAG_1, "value2")));

    assertThat(composed).isNotNull();
    assertThat(composed.isHost()).isTrue();
  }

  @Test
  public void compose_host_second() {
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(
            TransitionFactories.of(new StubPatch(FLAG_1, "value2")),
            HostTransition.createFactory());

    assertThat(composed).isNotNull();
    assertThat(composed.isHost()).isTrue();
  }

  @Test
  public void compose_noTrans_first() {
    TransitionFactory<StubData> patch = TransitionFactories.of(new StubPatch(FLAG_1, "value"));
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(NoTransition.createFactory(), patch);

    assertThat(composed).isNotNull();
    assertThat(composed).isEqualTo(patch);
  }

  @Test
  public void compose_noTrans_second() {
    TransitionFactory<StubData> patch = TransitionFactories.of(new StubPatch(FLAG_1, "value"));
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(patch, NoTransition.createFactory());

    assertThat(composed).isNotNull();
    assertThat(composed).isEqualTo(patch);
  }

  @Test
  public void compose_nullTrans_first() {
    StubPatch patch = new StubPatch(FLAG_1, "value");
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(
            NullTransition.createFactory(), TransitionFactories.of(patch));

    assertThat(composed).isNotNull();
    assertThat(NullTransition.isInstance(composed)).isTrue();
  }

  @Test
  public void compose_nullTrans_second() {
    StubPatch patch = new StubPatch(FLAG_1, "value");
    TransitionFactory<StubData> composed =
        ComposingTransitionFactory.of(
            TransitionFactories.of(patch), NullTransition.createFactory());

    assertThat(composed).isNotNull();
    assertThat(NullTransition.isInstance(composed)).isTrue();
  }

  private static final class StubData {}

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
    public BuildOptions patch(BuildOptions options) {
      return updateOptions(options, flagLabel, flagValue);
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
    public List<BuildOptions> split(BuildOptions options) {
      return flagValues.stream()
          .map(value -> updateOptions(options, flagLabel, value))
          .collect(toImmutableList());
    }
  }
}
