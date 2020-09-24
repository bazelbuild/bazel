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
package com.google.devtools.build.lib.testing.common;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FakeOptions} utility. */
@RunWith(JUnit4.class)
public class FakeOptionsTest {

  @Test
  public void getOptions_unspecifiedClass_returnsNull() {
    OptionsProvider optionsProvider = FakeOptions.builder().put(new TestOptions()).build();

    assertThat(optionsProvider.getOptions(TestOptions2.class)).isNull();
  }

  @Test
  public void getOptions_returnsProvidedValue() {
    TestOptions options = new TestOptions();
    options.value = "value";
    OptionsProvider optionsProvider = FakeOptions.builder().put(options).build();

    assertThat(optionsProvider.getOptions(TestOptions.class)).isEqualTo(options);
  }

  @Test
  public void getOptions_of_returnsOnlyProvidedValue() {
    TestOptions options = new TestOptions();
    options.value = "value";
    OptionsProvider optionsProvider = FakeOptions.of(options);

    assertThat(optionsProvider.getOptions(TestOptions.class)).isEqualTo(options);
    assertThat(optionsProvider.getOptions(TestOptions2.class)).isNull();
  }

  @Test
  public void getOptions_specifiedDefaultsClass_returnsDefaultOptions() {
    assertGetOptionsReturnsDefaults(
        FakeOptions.builder().putDefaults(TestOptions.class, TestOptions2.class).build());
  }

  @Test
  public void getOptions_ofDefaults_returnsDefaultOptions() {
    assertGetOptionsReturnsDefaults(FakeOptions.ofDefaults(TestOptions.class, TestOptions2.class));
  }

  private static void assertGetOptionsReturnsDefaults(OptionsProvider optionsProvider) {
    assertThat(optionsProvider.getOptions(TestOptions.class))
        .isEqualTo(Options.getDefaults(TestOptions.class));
    assertThat(optionsProvider.getOptions(TestOptions2.class))
        .isEqualTo(Options.getDefaults(TestOptions2.class));
  }

  @Test
  public void getStarlarkOptions_returnsEmpty() {
    OptionsProvider optionsProvider =
        FakeOptions.builder().put(new TestOptions()).putDefaults(TestOptions2.class).build();

    assertThat(optionsProvider.getStarlarkOptions()).isEmpty();
  }

  @Test
  public void getStarlarkOptions_emptyOptions_returnsEmpty() {
    assertThat(FakeOptions.builder().build().getStarlarkOptions()).isEmpty();
  }

  @Test
  public void build_specifiedValueTwiceForSameClass_fails() {
    FakeOptions.Builder builder =
        FakeOptions.builder().put(new TestOptions()).put(new TestOptions());

    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void build_specifiedValueAndDefaultsForSameClass_fails() {
    FakeOptions.Builder builder =
        FakeOptions.builder().put(new TestOptions()).putDefaults(TestOptions.class);

    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  public void build_defaultsTwiceForSameClass_fails() {
    FakeOptions.Builder builder =
        FakeOptions.builder().putDefaults(TestOptions.class, TestOptions.class);

    assertThrows(IllegalArgumentException.class, builder::build);
  }

  /** Simple test option class example. */
  public static final class TestOptions extends OptionsBase {
    @Option(
        name = "option1",
        defaultValue = "TestOptions default",
        effectTags = OptionEffectTag.NO_OP,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED)
    public String value;
  }

  /** Simple test option class, different from {@link TestOptions}. */
  public static final class TestOptions2 extends OptionsBase {
    @Option(
        name = "option2",
        defaultValue = "TestOptions2 default",
        effectTags = OptionEffectTag.NO_OP,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED)
    public String value;
  }
}
