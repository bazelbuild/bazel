// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import com.google.testing.junit.testparameterinjector.TestParameters.TestParametersValues;
import com.google.testing.junit.testparameterinjector.TestParametersValuesProvider;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link SimpleTargetPatternMatcher}. */
@RunWith(TestParameterInjector.class)
public class SimpleTargetPatternMatcherTest {
  @Test
  @TestParameters(valuesProvider = TargetPatternProvider.class)
  public void contains(boolean included, ImmutableList<String> patterns, Label label)
      throws Exception {
    SimpleTargetPatternMatcher matcher = SimpleTargetPatternMatcher.create(patterns);
    assertWithMessage("matcher %s contains %s", matcher, label)
        .that(matcher.contains(label))
        .isEqualTo(included);
  }

  @Test
  @TestParameters(valuesProvider = TargetPatternProvider.class)
  @SuppressWarnings("unused")
  public void toString(boolean included, ImmutableList<String> patterns, Label label)
      throws Exception {
    SimpleTargetPatternMatcher matcher = SimpleTargetPatternMatcher.create(patterns);
    String expected = String.format("[%s]", Joiner.on(",").join(patterns));
    assertThat(matcher.toString()).isEqualTo(expected);
  }

  static final class TargetPatternProvider extends TestParametersValuesProvider {
    private static TestParametersValues create(boolean included, String pattern, String label) {
      return create(included, ImmutableList.of(pattern), label);
    }

    private static TestParametersValues create(
        boolean included, List<String> patterns, String label) {
      String name = String.format("%s-%s-%s", included ? "included" : "excluded", patterns, label);
      return TestParametersValues.builder()
          .name(name)
          .addParameter("included", included)
          .addParameter("patterns", patterns)
          .addParameter("label", Label.parseCanonicalUnchecked(label))
          .build();
    }

    @Override
    protected ImmutableList<TestParametersValues> provideValues(Context context) {
      return ImmutableList.of(
          // Single pattern
          create(true, "//foo:foo", "//foo:foo"),
          create(true, "//foo:foo", "//foo"),
          create(true, "//foo", "//foo:foo"),
          create(true, "//foo", "//foo"),
          create(false, "//foo:foo", "//foo:bar"),
          create(false, "//foo", "//foo:bar"),
          create(true, "//foo/...", "//foo:foo"),
          create(true, "//foo/...", "//foo/bar:bar"),
          create(false, "//foo/...", "//bar:bar"),
          create(false, "//foo/bar/...", "//foo:foo"),
          create(false, "//foo", "//fooooooo"),
          create(false, "//foo/...", "//fooooooo"),

          // Multiple patterns
          create(true, ImmutableList.of("//foo:foo", "//bar:bar"), "//foo:foo"),
          create(true, ImmutableList.of("//foo:foo", "//bar:bar"), "//bar:bar"),
          create(false, ImmutableList.of("//foo:foo", "//bar:bar"), "//quux:quux"),

          // Negative patterns
          create(false, "-//foo:foo", "//foo:foo"),
          create(false, "-//foo/...", "//foo:foo"),
          create(false, ImmutableList.of("//foo/...", "-//foo/bar/..."), "//foo/bar:bar"),
          create(true, ImmutableList.of("//foo/...", "-//foo/bar/..."), "//foo:foo"),
          create(
              true,
              ImmutableList.of("//foo/...", "-//foo/bar/...", "//foo/bar/baz/..."),
              "//foo/bar/baz"),
          create(
              true,
              ImmutableList.of("//foo/...", "-//foo/bar/...", "//foo/bar/baz/..."),
              "//foo:foo"),
          create(
              false,
              ImmutableList.of("//foo/...", "-//foo/bar/...", "//foo/bar/baz/..."),
              "//foo/bar/quux"));
    }
  }
}
