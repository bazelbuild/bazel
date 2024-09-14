// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsTest;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ParsedFlagsValue}. */
@RunWith(JUnit4.class)
public final class ParsedFlagsValueTest {

  /** Extra options for this test. */
  public static final class DummyTestOptions extends FragmentOptions {

    @Option(
        name = "str_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defVal")
    public String strOption;

    @Option(
        name = "another_str_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defVal")
    public String anotherStrOption;

    @Option(
        name = "bool_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "false")
    public boolean boolOption;

    @Option(
        name = "list_option",
        converter = CommaSeparatedOptionListConverter.class,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "null")
    public List<String> listOption;

    @Option(
        name = "null_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "null")
    public String nullOption;

    @Option(
        name = "accumulating_option",
        allowMultiple = true,
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "null")
    public List<String> accumulatingOption;

    @Option(
        name = "dummy_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "internal_default",
        implicitRequirements = {"--implicit_option=set_implicitly"})
    public String dummyOption;

    @Option(
        name = "implicit_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "implicit_default")
    public String implicitOption;
  }

  private static final ImmutableSet<Class<? extends FragmentOptions>> BUILD_CONFIG_OPTIONS =
      ImmutableSet.of(DummyTestOptions.class);

  @Test
  public void parse() throws Exception {
    NativeAndStarlarkFlags flags =
        NativeAndStarlarkFlags.builder()
            .optionsClasses(BUILD_CONFIG_OPTIONS)
            .nativeFlags(ImmutableList.of("--str_option=bar", "--nobool_option"))
            .starlarkFlags(ImmutableMap.of("//custom:flag", "hello"))
            .starlarkFlagDefaults(ImmutableMap.of("//custom:flag", "default"))
            .build();

    ParsedFlagsValue parsedFlags = ParsedFlagsValue.parseAndCreate(flags);

    OptionsParsingResult result = parsedFlags.parsingResult();
    assertThat(result.getOptions(DummyTestOptions.class).strOption).isEqualTo("bar");
    assertThat(result.getOptions(DummyTestOptions.class).boolOption).isFalse();
    assertThat(result.getStarlarkOptions()).containsAtLeast("//custom:flag", "hello");
  }

  @Test
  public void mergeWith() throws Exception {
    BuildOptions original =
        BuildOptions.of(BUILD_CONFIG_OPTIONS, "--str_option=foo", "--bool_option");

    NativeAndStarlarkFlags flags =
        NativeAndStarlarkFlags.builder()
            .optionsClasses(BUILD_CONFIG_OPTIONS)
            .nativeFlags(ImmutableList.of("--str_option=bar", "--nobool_option"))
            .starlarkFlags(ImmutableMap.of("//custom:flag", "hello"))
            .starlarkFlagDefaults(ImmutableMap.of("//custom:flag", "default"))
            .build();
    ParsedFlagsValue parsedFlags = ParsedFlagsValue.parseAndCreate(flags);

    BuildOptions modified = parsedFlags.mergeWith(original).getOptions();

    // Ensure the original wasn't modified.
    assertThat(original.get(DummyTestOptions.class))
        .isNotEqualTo(modified.get(DummyTestOptions.class));

    // Check the modified values.
    assertThat(modified.get(DummyTestOptions.class).strOption).isEqualTo("bar");
    assertThat(modified.get(DummyTestOptions.class).boolOption).isFalse();
    assertThat(modified.getStarlarkOptions())
        .containsAtLeast(Label.parseCanonicalUnchecked("//custom:flag"), "hello");
  }

  @Test
  public void mergeWith_unknownNativeFragment() throws Exception {
    // Only use the basic flags.
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    // Add another fragment with different flags.
    NativeAndStarlarkFlags flags =
        NativeAndStarlarkFlags.builder()
            .optionsClasses(
                ImmutableSet.<Class<? extends FragmentOptions>>builder()
                    .addAll(BUILD_CONFIG_OPTIONS)
                    .add(BuildOptionsTest.SecondDummyTestOptions.class)
                    .build())
            .nativeFlags(ImmutableList.of("--second_str_option=bar"))
            .build();
    ParsedFlagsValue parsedFlags = ParsedFlagsValue.parseAndCreate(flags);

    // The native flags that are unknown to the original options should not be present.
    BuildOptions modified = parsedFlags.mergeWith(original).getOptions();
    assertThat(modified.contains(BuildOptionsTest.SecondDummyTestOptions.class)).isFalse();
  }

  @Test
  public void mergeWith_illegalStarlarkLabel() throws Exception {
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    NativeAndStarlarkFlags flags =
        NativeAndStarlarkFlags.builder()
            .optionsClasses(BUILD_CONFIG_OPTIONS)
            .starlarkFlags(ImmutableMap.of("@@@", "hello"))
            .build();
    ParsedFlagsValue parsedFlagsValue = ParsedFlagsValue.parseAndCreate(flags);

    // BuildOptions, unlike OptionsParser, uses a Label for the key, so this is the only code path
    // that validates that a starlark flag is actually a Label.
    assertThrows(IllegalArgumentException.class, () -> parsedFlagsValue.mergeWith(original));
  }

  @Test
  public void mergeWith_multiValueOption_nonAccumulating() throws Exception {
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS, "--list_option=baz,quux");

    NativeAndStarlarkFlags flags =
        NativeAndStarlarkFlags.builder()
            .optionsClasses(BUILD_CONFIG_OPTIONS)
            .nativeFlags(ImmutableList.of("--list_option=foo,bar"))
            .build();
    ParsedFlagsValue parsedFlags = ParsedFlagsValue.parseAndCreate(flags);

    BuildOptions modified = parsedFlags.mergeWith(original).getOptions();

    assertThat(modified.get(DummyTestOptions.class).listOption)
        // Because this flag does not allow multiple values the list simply overwrites the previous
        // value.
        .containsExactly("foo", "bar")
        .inOrder();
  }

  @Test
  public void mergeWith_implicitOption() throws Exception {
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    NativeAndStarlarkFlags flags =
        NativeAndStarlarkFlags.builder()
            .optionsClasses(BUILD_CONFIG_OPTIONS)
            .nativeFlags(ImmutableList.of("--dummy_option=direct"))
            .build();
    ParsedFlagsValue parsedFlags = ParsedFlagsValue.parseAndCreate(flags);

    BuildOptions modified = parsedFlags.mergeWith(original).getOptions();

    assertThat(modified.get(DummyTestOptions.class).dummyOption).isEqualTo("direct");
    assertThat(modified.get(DummyTestOptions.class).implicitOption).isEqualTo("set_implicitly");
  }

  @Test
  public void mergeWith_accumulating() throws Exception {
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    NativeAndStarlarkFlags flags =
        NativeAndStarlarkFlags.builder()
            .optionsClasses(BUILD_CONFIG_OPTIONS)
            .nativeFlags(ImmutableList.of("--accumulating_option=foo", "--accumulating_option=bar"))
            .build();
    ParsedFlagsValue parsedFlags = ParsedFlagsValue.parseAndCreate(flags);

    BuildOptions modified = parsedFlags.mergeWith(original).getOptions();

    assertThat(modified.get(DummyTestOptions.class).accumulatingOption)
        .containsExactly("foo", "bar")
        .inOrder();
  }

  // TODO: https://github.com/bazelbuild/bazel/issues/22453 - Add a test of an accumulating flag
  // with previous values when that works correctly.

  @Test
  public void mergeWith_starlark() throws Exception {
    BuildOptions original =
        BuildOptions.of(BUILD_CONFIG_OPTIONS).toBuilder()
            .addStarlarkOption(Label.parseCanonicalUnchecked("//custom:flag"), "direct")
            .build();

    NativeAndStarlarkFlags flags =
        NativeAndStarlarkFlags.builder()
            .optionsClasses(BUILD_CONFIG_OPTIONS)
            .starlarkFlags(ImmutableMap.of("//custom:flag", "override"))
            .starlarkFlagDefaults(ImmutableMap.of("//custom:flag", "default"))
            .build();
    ParsedFlagsValue parsedFlags = ParsedFlagsValue.parseAndCreate(flags);

    BuildOptions modified = parsedFlags.mergeWith(original).getOptions();

    // Check the modified values.
    assertThat(modified.getStarlarkOptions())
        .containsAtLeast(Label.parseCanonicalUnchecked("//custom:flag"), "override");
  }

  @Test
  public void mergeWith_starlark_resetToDefault() throws Exception {
    BuildOptions original =
        BuildOptions.of(BUILD_CONFIG_OPTIONS).toBuilder()
            .addStarlarkOption(Label.parseCanonicalUnchecked("//custom:flag"), "direct")
            .build();

    NativeAndStarlarkFlags flags =
        NativeAndStarlarkFlags.builder()
            .optionsClasses(BUILD_CONFIG_OPTIONS)
            .starlarkFlags(ImmutableMap.of("//custom:flag", "default"))
            .starlarkFlagDefaults(ImmutableMap.of("//custom:flag", "default"))
            .build();
    ParsedFlagsValue parsedFlags = ParsedFlagsValue.parseAndCreate(flags);

    BuildOptions modified = parsedFlags.mergeWith(original).getOptions();

    // The Starlark flag should not be present since it was reset to the default value
    assertThat(modified.getStarlarkOptions())
        .doesNotContainKey(Label.parseCanonicalUnchecked("//custom:flag"));
  }
}
