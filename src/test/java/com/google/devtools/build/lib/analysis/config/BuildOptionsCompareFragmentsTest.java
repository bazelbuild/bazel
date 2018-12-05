// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsDiffForReconstruction;
import com.google.devtools.build.lib.skyframe.trimming.ConfigurationComparer;
import com.google.devtools.build.lib.skyframe.trimming.TrimmableTestConfigurationFragments.AOptions;
import com.google.devtools.build.lib.skyframe.trimming.TrimmableTestConfigurationFragments.BOptions;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import java.util.Objects;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Tests of compareFragments in BuildOptions.OptionsDiffForReconstruction. */
public final class BuildOptionsCompareFragmentsTest {

  /** Test cases for BuildOptionsCompareFragmentsTest. */
  @AutoValue
  public abstract static class Case {
    public abstract String getName();

    public abstract BuildOptions getBase();

    public abstract BuildOptions getLeft();

    public abstract BuildOptions getRight();

    public abstract ConfigurationComparer.Result getLeftToRightResult();

    public abstract ConfigurationComparer.Result getRightToLeftResult();

    public static Builder named(String name) {
      return new AutoValue_BuildOptionsCompareFragmentsTest_Case.Builder().setName(name);
    }

    /** Quick builder for test cases. */
    @AutoValue.Builder
    public abstract static class Builder {
      public abstract Builder setName(String name);

      public abstract Builder setBase(BuildOptions base);

      public Builder setBase(OptionsBuilder base) throws Exception {
        return this.setBase(base.build());
      }

      public abstract Builder setLeft(BuildOptions left);

      public Builder setLeft(OptionsBuilder left) throws Exception {
        return this.setLeft(left.build());
      }

      public abstract Builder setRight(BuildOptions right);

      public Builder setRight(OptionsBuilder right) throws Exception {
        return this.setRight(right.build());
      }

      public Builder setResult(ConfigurationComparer.Result result) {
        return this.setLeftToRightResult(result).setRightToLeftResult(result);
      }

      public abstract Builder setLeftToRightResult(ConfigurationComparer.Result result);

      public abstract Builder setRightToLeftResult(ConfigurationComparer.Result result);

      public abstract Case build();
    }

    @Override
    public final String toString() {
      if (Objects.equals(this.getLeftToRightResult(), this.getRightToLeftResult())) {
        return String.format("%s [result = %s]", this.getName(), this.getLeftToRightResult());
      } else {
        return String.format(
            "%s [compare(left, right) = %s; compare(right, left) = %s]",
            this.getName(), this.getLeftToRightResult(), this.getRightToLeftResult());
      }
    }
  }

  /** Quick builder for BuildOptions instances. */
  public static final class OptionsBuilder {
    private final ImmutableMap.Builder<String, Object> starlarkOptions =
        new ImmutableMap.Builder<>();
    private final ImmutableList.Builder<Class<? extends FragmentOptions>> fragments =
        new ImmutableList.Builder<>();
    private final ImmutableList.Builder<String> nativeOptions = new ImmutableList.Builder<>();

    public OptionsBuilder withNativeFragment(
        Class<? extends FragmentOptions> fragment, String... flags) {
      this.fragments.add(fragment);
      this.nativeOptions.add(flags);
      return this;
    }

    public OptionsBuilder withStarlarkOption(String setting, Object value) {
      this.starlarkOptions.put(setting, value);
      return this;
    }

    public OptionsBuilder withStarlarkOption(String setting) {
      return this.withStarlarkOption(setting, setting);
    }

    public BuildOptions build() throws Exception {
      return BuildOptions.builder()
          .addStarlarkOptions(this.starlarkOptions.build())
          .merge(
              BuildOptions.of(
                  this.fragments.build(), this.nativeOptions.build().toArray(new String[0])))
          .build();
    }
  }

  /** Test cases for compareFragments which produce an ordinary result. */
  @RunWith(Parameterized.class)
  public static final class NormalCases {

    @Parameters(name = "{index}: {0}")
    public static Iterable<Case> cases() throws Exception {
      return ImmutableList.of(
          Case.named("both options equal to the base")
              .setResult(ConfigurationComparer.Result.EQUAL)
              .setBase(
                  new OptionsBuilder()
                      .withNativeFragment(AOptions.class)
                      .withStarlarkOption("//alpha"))
              .setLeft(
                  new OptionsBuilder()
                      .withNativeFragment(AOptions.class)
                      .withStarlarkOption("//alpha"))
              .setRight(
                  new OptionsBuilder()
                      .withNativeFragment(AOptions.class)
                      .withStarlarkOption("//alpha"))
              .build(),
          Case.named("both sides change native fragment to same value")
              .setResult(ConfigurationComparer.Result.EQUAL)
              .setBase(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=base"))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=new"))
              .setRight(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=new"))
              .build(),
          Case.named("both sides add native fragment with same value")
              .setResult(ConfigurationComparer.Result.EQUAL)
              .setBase(new OptionsBuilder())
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=new"))
              .setRight(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=new"))
              .build(),
          Case.named("both sides remove same native fragment")
              .setResult(ConfigurationComparer.Result.EQUAL)
              .setBase(new OptionsBuilder().withNativeFragment(AOptions.class))
              .setLeft(new OptionsBuilder())
              .setRight(new OptionsBuilder())
              .build(),
          Case.named("both sides change Starlark option to same value")
              .setResult(ConfigurationComparer.Result.EQUAL)
              .setBase(new OptionsBuilder().withStarlarkOption("//alpha", "base"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha", "new"))
              .setRight(new OptionsBuilder().withStarlarkOption("//alpha", "new"))
              .build(),
          Case.named("both sides add Starlark option with same value")
              .setResult(ConfigurationComparer.Result.EQUAL)
              .setBase(new OptionsBuilder())
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha", "new"))
              .setRight(new OptionsBuilder().withStarlarkOption("//alpha", "new"))
              .build(),
          Case.named("both sides remove same Starlark option")
              .setResult(ConfigurationComparer.Result.EQUAL)
              .setBase(new OptionsBuilder().withStarlarkOption("//alpha"))
              .setLeft(new OptionsBuilder())
              .setRight(new OptionsBuilder())
              .build(),
          Case.named("native fragment removed from right")
              .setLeftToRightResult(ConfigurationComparer.Result.SUPERSET)
              .setRightToLeftResult(ConfigurationComparer.Result.SUBSET)
              .setBase(new OptionsBuilder().withNativeFragment(AOptions.class))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class))
              .setRight(new OptionsBuilder())
              .build(),
          Case.named("native fragment added to right")
              .setLeftToRightResult(ConfigurationComparer.Result.SUBSET)
              .setRightToLeftResult(ConfigurationComparer.Result.SUPERSET)
              .setBase(new OptionsBuilder())
              .setLeft(new OptionsBuilder())
              .setRight(new OptionsBuilder().withNativeFragment(AOptions.class))
              .build(),
          Case.named("native fragment changed in left and removed from right")
              .setLeftToRightResult(ConfigurationComparer.Result.SUPERSET)
              .setRightToLeftResult(ConfigurationComparer.Result.SUBSET)
              .setBase(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=base"))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=left"))
              .setRight(new OptionsBuilder())
              .build(),
          Case.named("native fragment added to left and another fragment removed from right")
              .setLeftToRightResult(ConfigurationComparer.Result.SUPERSET)
              .setRightToLeftResult(ConfigurationComparer.Result.SUBSET)
              .setBase(new OptionsBuilder().withNativeFragment(AOptions.class))
              .setLeft(
                  new OptionsBuilder()
                      .withNativeFragment(AOptions.class)
                      .withNativeFragment(BOptions.class))
              .setRight(new OptionsBuilder())
              .build(),
          Case.named("Starlark option removed from right")
              .setLeftToRightResult(ConfigurationComparer.Result.SUPERSET)
              .setRightToLeftResult(ConfigurationComparer.Result.SUBSET)
              .setBase(new OptionsBuilder().withStarlarkOption("//alpha"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha"))
              .setRight(new OptionsBuilder())
              .build(),
          Case.named("Starlark option added to right")
              .setLeftToRightResult(ConfigurationComparer.Result.SUBSET)
              .setRightToLeftResult(ConfigurationComparer.Result.SUPERSET)
              .setBase(new OptionsBuilder())
              .setLeft(new OptionsBuilder())
              .setRight(new OptionsBuilder().withStarlarkOption("//alpha"))
              .build(),
          Case.named("Starlark option changed in left and removed from right")
              .setLeftToRightResult(ConfigurationComparer.Result.SUPERSET)
              .setRightToLeftResult(ConfigurationComparer.Result.SUBSET)
              .setBase(new OptionsBuilder().withStarlarkOption("//alpha", "base"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha", "left"))
              .setRight(new OptionsBuilder())
              .build(),
          Case.named("Starlark option added to left and another option removed from right")
              .setLeftToRightResult(ConfigurationComparer.Result.SUPERSET)
              .setRightToLeftResult(ConfigurationComparer.Result.SUBSET)
              .setBase(new OptionsBuilder().withStarlarkOption("//alpha"))
              .setLeft(
                  new OptionsBuilder().withStarlarkOption("//alpha").withStarlarkOption("//bravo"))
              .setRight(new OptionsBuilder())
              .build(),
          Case.named("different native fragment added to each side")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(new OptionsBuilder())
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class))
              .setRight(new OptionsBuilder().withNativeFragment(BOptions.class))
              .build(),
          Case.named("different native fragment removed from each side")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(
                  new OptionsBuilder()
                      .withNativeFragment(AOptions.class)
                      .withNativeFragment(BOptions.class))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class))
              .setRight(new OptionsBuilder().withNativeFragment(BOptions.class))
              .build(),
          Case.named("native fragment added and different fragment removed on left")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(new OptionsBuilder().withNativeFragment(BOptions.class))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class))
              .setRight(new OptionsBuilder().withNativeFragment(BOptions.class))
              .build(),
          Case.named(
                  "native fragment added to right; "
                      + "other fragment changed on left and removed from right")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=base"))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=left"))
              .setRight(new OptionsBuilder().withNativeFragment(BOptions.class))
              .build(),
          Case.named("native fragment changed on each side, removed from the other")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(
                  new OptionsBuilder()
                      .withNativeFragment(AOptions.class, "--alpha=base")
                      .withNativeFragment(BOptions.class, "--bravo=base"))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=left"))
              .setRight(new OptionsBuilder().withNativeFragment(BOptions.class, "--bravo=right"))
              .build(),
          Case.named(
                  "native fragment changed on left, removed from right; "
                      + "other fragment removed from left")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(
                  new OptionsBuilder()
                      .withNativeFragment(AOptions.class, "--alpha=base")
                      .withNativeFragment(BOptions.class))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=left"))
              .setRight(new OptionsBuilder().withNativeFragment(BOptions.class))
              .build(),
          Case.named("different Starlark option added to each side")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(new OptionsBuilder())
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha"))
              .setRight(new OptionsBuilder().withStarlarkOption("//bravo"))
              .build(),
          Case.named("different Starlark option removed from each side")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(
                  new OptionsBuilder().withStarlarkOption("//alpha").withStarlarkOption("//bravo"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha"))
              .setRight(new OptionsBuilder().withStarlarkOption("//bravo"))
              .build(),
          Case.named("Starlark option added and different option removed on left")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(new OptionsBuilder().withStarlarkOption("//bravo"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha"))
              .setRight(new OptionsBuilder().withStarlarkOption("//bravo"))
              .build(),
          Case.named(
                  "Starlark option added to right; "
                      + "other option changed on left and removed from right")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(new OptionsBuilder().withStarlarkOption("//alpha", "base"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha", "left"))
              .setRight(new OptionsBuilder().withStarlarkOption("//bravo"))
              .build(),
          Case.named("Starlark option changed on each side, removed from the other")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(
                  new OptionsBuilder()
                      .withStarlarkOption("//alpha", "base")
                      .withStarlarkOption("//bravo", "base"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha", "left"))
              .setRight(new OptionsBuilder().withStarlarkOption("//bravo", "right"))
              .build(),
          Case.named(
                  "Starlark option changed on left, removed from right; "
                      + "other option removed from left")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(
                  new OptionsBuilder()
                      .withStarlarkOption("//alpha", "base")
                      .withStarlarkOption("//bravo"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha", "left"))
              .setRight(new OptionsBuilder().withStarlarkOption("//bravo"))
              .build(),
          Case.named("Starlark option removed from left, native option removed from right")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(
                  new OptionsBuilder()
                      .withNativeFragment(AOptions.class)
                      .withStarlarkOption("//bravo"))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class))
              .setRight(new OptionsBuilder().withStarlarkOption("//bravo"))
              .build(),
          Case.named("Starlark option added to left, native option added to right")
              .setResult(ConfigurationComparer.Result.ALL_SHARED_FRAGMENTS_EQUAL)
              .setBase(new OptionsBuilder())
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha"))
              .setRight(new OptionsBuilder().withNativeFragment(BOptions.class))
              .build(),
          Case.named("native fragment is unchanged in left, changes in right")
              .setResult(ConfigurationComparer.Result.DIFFERENT)
              .setBase(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=base"))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=base"))
              .setRight(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=right"))
              .build(),
          Case.named("native fragment is changed to different values")
              .setResult(ConfigurationComparer.Result.DIFFERENT)
              .setBase(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=base"))
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=left"))
              .setRight(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=right"))
              .build(),
          Case.named("native fragment is added with different values")
              .setResult(ConfigurationComparer.Result.DIFFERENT)
              .setBase(new OptionsBuilder())
              .setLeft(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=left"))
              .setRight(new OptionsBuilder().withNativeFragment(AOptions.class, "--alpha=right"))
              .build(),
          Case.named("Starlark option is unchanged in left, changes in right")
              .setResult(ConfigurationComparer.Result.DIFFERENT)
              .setBase(new OptionsBuilder().withStarlarkOption("//alpha", "base"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha", "base"))
              .setRight(new OptionsBuilder().withStarlarkOption("//alpha", "right"))
              .build(),
          Case.named("Starlark option is changed to different values")
              .setResult(ConfigurationComparer.Result.DIFFERENT)
              .setBase(new OptionsBuilder().withStarlarkOption("//alpha", "base"))
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha", "left"))
              .setRight(new OptionsBuilder().withStarlarkOption("//alpha", "right"))
              .build(),
          Case.named("Starlark option is added with different values")
              .setResult(ConfigurationComparer.Result.DIFFERENT)
              .setBase(new OptionsBuilder())
              .setLeft(new OptionsBuilder().withStarlarkOption("//alpha", "left"))
              .setRight(new OptionsBuilder().withStarlarkOption("//alpha", "right"))
              .build());
    }

    private final Case testCase;

    public NormalCases(Case testCase) {
      this.testCase = testCase;
    }

    @Test
    public void compareLeftToRight() throws Exception {
      OptionsDiffForReconstruction diffLeft =
          BuildOptions.diffForReconstruction(testCase.getBase(), testCase.getLeft());
      OptionsDiffForReconstruction diffRight =
          BuildOptions.diffForReconstruction(testCase.getBase(), testCase.getRight());

      assertThat(OptionsDiffForReconstruction.compareFragments(diffLeft, diffRight))
          .isEqualTo(testCase.getLeftToRightResult());
    }

    @Test
    public void compareRightToLeft() throws Exception {
      OptionsDiffForReconstruction diffLeft =
          BuildOptions.diffForReconstruction(testCase.getBase(), testCase.getLeft());
      OptionsDiffForReconstruction diffRight =
          BuildOptions.diffForReconstruction(testCase.getBase(), testCase.getRight());

      assertThat(OptionsDiffForReconstruction.compareFragments(diffRight, diffLeft))
          .isEqualTo(testCase.getRightToLeftResult());
    }
  }

  /** Test cases for compareFragments which produce errors. */
  @RunWith(JUnit4.class)
  public static final class ExceptionalCases {
    @Test
    public void withDifferentBases_throwsError() throws Exception {
      BuildOptions baseA =
          new OptionsBuilder()
              .withNativeFragment(AOptions.class, "--alpha=A")
              .withNativeFragment(BOptions.class, "--bravo=base")
              .build();
      BuildOptions newA =
          new OptionsBuilder()
              .withNativeFragment(AOptions.class, "--alpha=A")
              .withNativeFragment(BOptions.class, "--bravo=new")
              .build();
      BuildOptions baseB =
          new OptionsBuilder()
              .withNativeFragment(AOptions.class, "--alpha=B")
              .withNativeFragment(BOptions.class, "--bravo=base")
              .build();
      BuildOptions newB =
          new OptionsBuilder()
              .withNativeFragment(AOptions.class, "--alpha=B")
              .withNativeFragment(BOptions.class, "--bravo=old")
              .build();

      OptionsDiffForReconstruction diffA = BuildOptions.diffForReconstruction(baseA, newA);
      OptionsDiffForReconstruction diffB = BuildOptions.diffForReconstruction(baseB, newB);

      IllegalArgumentException forwardException =
          MoreAsserts.assertThrows(
              IllegalArgumentException.class,
              () -> OptionsDiffForReconstruction.compareFragments(diffA, diffB));
      assertThat(forwardException).hasMessageThat().contains("diffs with different bases");

      IllegalArgumentException reverseException =
          MoreAsserts.assertThrows(
              IllegalArgumentException.class,
              () -> OptionsDiffForReconstruction.compareFragments(diffB, diffA));
      assertThat(reverseException).hasMessageThat().contains("diffs with different bases");
    }
  }
}
