// Copyright 2009 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableClassToInstanceMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.BuildOptions.MapBackedChecksumCache;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsChecksumCache;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodecs;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsParser;
import com.google.protobuf.ByteString;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link BuildOptions}.
 *
 * <p>Currently this tests native options and Starlark options completely separately since these two
 * types of options do not interact. In the future when we begin to migrate native options to
 * Starlark options, the format of this test class will need to accommodate that overlap.
 */
@RunWith(JUnit4.class)
public final class BuildOptionsTest {

  /** Extra options for this test. */
  public static class DummyTestOptions extends FragmentOptions {
    public DummyTestOptions() {}

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
  }

  /** Extra options for this test. */
  public static class SecondDummyTestOptions extends FragmentOptions {
    @Option(
        name = "second_str_option",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "defVal")
    public String strOption;
  }

  private static final ImmutableList<Class<? extends FragmentOptions>> BUILD_CONFIG_OPTIONS =
      ImmutableList.of(DummyTestOptions.class);

  @Test
  public void optionSetCaching() {
    BuildOptions a =
        BuildOptions.of(
            BUILD_CONFIG_OPTIONS,
            OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build());
    BuildOptions b =
        BuildOptions.of(
            BUILD_CONFIG_OPTIONS,
            OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build());
    // The cache keys of the OptionSets must be equal even if these are
    // different objects, if they were created with the same options (no options in this case).
    assertThat(b.toString()).isEqualTo(a.toString());
    assertThat(b.checksum()).isEqualTo(a.checksum());
    assertThat(a).isEqualTo(b);
  }

  @Test
  public void optionsEquality() throws Exception {
    String[] options1 = new String[] {"--str_option=foo"};
    String[] options2 = new String[] {"--str_option=bar"};
    // Distinct instances with the same values are equal:
    assertThat(BuildOptions.of(BUILD_CONFIG_OPTIONS, options1))
        .isEqualTo(BuildOptions.of(BUILD_CONFIG_OPTIONS, options1));
    // Same fragments, different values aren't equal:
    assertThat(
            BuildOptions.of(BUILD_CONFIG_OPTIONS, options1)
                .equals(BuildOptions.of(BUILD_CONFIG_OPTIONS, options2)))
        .isFalse();
    // Same values, different fragments aren't equal:
    assertThat(
            BuildOptions.of(BUILD_CONFIG_OPTIONS, options1)
                .equals(
                    BuildOptions.of(
                        ImmutableList.of(DummyTestOptions.class, SecondDummyTestOptions.class),
                        options1)))
        .isFalse();
  }

  @Test
  public void serialization() throws Exception {
    new SerializationTester(
            BuildOptions.of(makeOptionsClassBuilder().build(), "--str_option=foo"),
            BuildOptions.of(makeOptionsClassBuilder().build(), "--str_option=bar"),
            BuildOptions.of(makeOptionsClassBuilder().add(SecondDummyTestOptions.class).build()),
            BuildOptions.of(
                makeOptionsClassBuilder().add(SecondDummyTestOptions.class).build(),
                "--str_option=foo",
                "--second_str_option=baz",
                "--another_str_option=bar"))
        .addDependency(OptionsChecksumCache.class, new MapBackedChecksumCache())
        .runTests();
  }

  private static ImmutableList.Builder<Class<? extends FragmentOptions>> makeOptionsClassBuilder() {
    return ImmutableList.<Class<? extends FragmentOptions>>builder().addAll(BUILD_CONFIG_OPTIONS);
  }

  @Test
  public void serialize_primeFails_throws() throws Exception {
    OptionsChecksumCache failToPrimeCache =
        new OptionsChecksumCache() {
          @Override
          public BuildOptions getOptions(String checksum) {
            throw new UnsupportedOperationException();
          }

          @Override
          public boolean prime(BuildOptions options) {
            return false;
          }
        };
    BuildOptions options = BuildOptions.of(BUILD_CONFIG_OPTIONS);
    ObjectCodecs codecs =
        new ObjectCodecs(
            ImmutableClassToInstanceMap.of(OptionsChecksumCache.class, failToPrimeCache));
    assertThrows(SerializationException.class, () -> codecs.serialize(options));
  }

  @Test
  public void deserialize_unprimedCache_throws() throws Exception {
    BuildOptions options = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    ObjectCodecs codecs =
        new ObjectCodecs(
            ImmutableClassToInstanceMap.of(
                OptionsChecksumCache.class, new MapBackedChecksumCache()));
    ByteString bytes = codecs.serialize(options);
    assertThat(bytes).isNotNull();

    // Different checksum cache than the one used for serialization, and it has not been primed.
    ObjectCodecs notPrimed =
        new ObjectCodecs(
            ImmutableClassToInstanceMap.of(
                OptionsChecksumCache.class, new MapBackedChecksumCache()));
    Exception e = assertThrows(SerializationException.class, () -> notPrimed.deserialize(bytes));
    assertThat(e).hasMessageThat().contains(options.checksum());
  }

  @Test
  public void deserialize_primedCache_returnsPrimedInstance() throws Exception {
    BuildOptions options = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    ObjectCodecs codecs =
        new ObjectCodecs(
            ImmutableClassToInstanceMap.of(
                OptionsChecksumCache.class, new MapBackedChecksumCache()));
    ByteString bytes = codecs.serialize(options);
    assertThat(bytes).isNotNull();

    // Different checksum cache than the one used for serialization, but it has been primed.
    OptionsChecksumCache checksumCache = new MapBackedChecksumCache();
    assertThat(checksumCache.prime(options)).isTrue();
    ObjectCodecs primed =
        new ObjectCodecs(ImmutableClassToInstanceMap.of(OptionsChecksumCache.class, checksumCache));
    assertThat(primed.deserialize(bytes)).isSameInstanceAs(options);
  }

  @Test
  public void testMultiValueOptionImmutability() {
    BuildOptions options =
        BuildOptions.of(
            BUILD_CONFIG_OPTIONS,
            OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build());
    DummyTestOptions dummyTestOptions = options.get(DummyTestOptions.class);
    assertThrows(
        UnsupportedOperationException.class, () -> dummyTestOptions.accumulatingOption.add("foo"));
  }

  @Test
  public void parsingResultTransform() throws Exception {
    BuildOptions original =
        BuildOptions.of(BUILD_CONFIG_OPTIONS, "--str_option=foo", "--bool_option");

    OptionsParser parser = OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build();
    parser.parse("--str_option=bar", "--nobool_option");
    parser.setStarlarkOptions(ImmutableMap.of("//custom:flag", "hello"));

    BuildOptions modified = original.applyParsingResult(parser);

    assertThat(original.get(DummyTestOptions.class).strOption)
        .isNotEqualTo(modified.get(DummyTestOptions.class).strOption);
    assertThat(modified.get(DummyTestOptions.class).strOption).isEqualTo("bar");
    assertThat(modified.get(DummyTestOptions.class).boolOption).isFalse();
    assertThat(modified.getStarlarkOptions().get(Label.parseCanonicalUnchecked("//custom:flag")))
        .isEqualTo("hello");
  }

  @Test
  public void parsingResultTransformNativeIgnored() throws Exception {
    // Only use the basic flags.
    BuildOptions original = BuildOptions.of(makeOptionsClassBuilder().build());

    // Add another fragment with different flags.
    OptionsParser parser =
        OptionsParser.builder()
            .optionsClasses(makeOptionsClassBuilder().add(SecondDummyTestOptions.class).build())
            .build();
    parser.parse("--second_str_option=bar");

    // The flags that are unknown to the original options should not be present.
    BuildOptions modified = original.applyParsingResult(parser);
    assertThat(modified.contains(SecondDummyTestOptions.class)).isFalse();
  }

  @Test
  public void parsingResultTransformIllegalStarlarkLabel() throws Exception {
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    OptionsParser parser = OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build();
    parser.setStarlarkOptions(ImmutableMap.of("@@@", "hello"));

    assertThrows(IllegalArgumentException.class, () -> original.applyParsingResult(parser));
  }

  @Test
  public void parsingResultTransformMultiValueOption() throws Exception {
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    OptionsParser parser = OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build();
    parser.parse("--list_option=foo");

    BuildOptions modified = original.applyParsingResult(parser);

    assertThat(modified.get(DummyTestOptions.class).listOption).containsExactly("foo");
  }

  @Test
  public void parsingResultMatch() throws Exception {
    BuildOptions original =
        BuildOptions.of(BUILD_CONFIG_OPTIONS, "--str_option=foo", "--bool_option");

    OptionsParser matchingParser =
        OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build();
    matchingParser.parse("--str_option=foo", "--bool_option");

    OptionsParser notMatchingParser =
        OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build();
    notMatchingParser.parse("--str_option=foo", "--nobool_option");

    assertThat(original.matches(matchingParser)).isTrue();
    assertThat(original.matches(notMatchingParser)).isFalse();
  }

  @Test
  public void parsingResultMatchStarlark() throws Exception {
    BuildOptions original =
        BuildOptions.builder()
            .addStarlarkOption(Label.parseCanonicalUnchecked("//custom:flag"), "hello")
            .build();

    OptionsParser matchingParser =
        OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build();
    matchingParser.setStarlarkOptions(ImmutableMap.of("//custom:flag", "hello"));

    OptionsParser notMatchingParser =
        OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build();
    notMatchingParser.setStarlarkOptions(ImmutableMap.of("//custom:flag", "foo"));

    assertThat(original.matches(matchingParser)).isTrue();
    assertThat(original.matches(notMatchingParser)).isFalse();
  }

  @Test
  public void parsingResultMatchMissingFragment() throws Exception {
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS, "--str_option=foo");

    ImmutableList<Class<? extends FragmentOptions>> fragmentClasses =
        ImmutableList.of(DummyTestOptions.class, SecondDummyTestOptions.class);

    OptionsParser parser = OptionsParser.builder().optionsClasses(fragmentClasses).build();
    parser.parse("--str_option=foo", "--second_str_option=bar");

    assertThat(original.matches(parser)).isTrue();
  }

  @Test
  public void parsingResultMatchEmptyNativeMatch() throws Exception {
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS, "--str_option=foo");

    ImmutableList<Class<? extends FragmentOptions>> fragmentClasses =
        ImmutableList.of(DummyTestOptions.class, SecondDummyTestOptions.class);

    OptionsParser parser = OptionsParser.builder().optionsClasses(fragmentClasses).build();
    parser.parse("--second_str_option=bar");

    assertThat(original.matches(parser)).isFalse();
  }

  @Test
  public void parsingResultMatchEmptyNativeMatchWithStarlark() throws Exception {
    BuildOptions original =
        BuildOptions.builder()
            .addStarlarkOption(Label.parseCanonicalUnchecked("//custom:flag"), "hello")
            .build();

    ImmutableList<Class<? extends FragmentOptions>> fragmentClasses =
        ImmutableList.<Class<? extends FragmentOptions>>builder()
            .add(DummyTestOptions.class)
            .add(SecondDummyTestOptions.class)
            .build();

    OptionsParser parser = OptionsParser.builder().optionsClasses(fragmentClasses).build();
    parser.parse("--second_str_option=bar");
    parser.setStarlarkOptions(ImmutableMap.of("//custom:flag", "hello"));

    assertThat(original.matches(parser)).isTrue();
  }

  @Test
  public void parsingResultMatchStarlarkOptionMissing() throws Exception {
    BuildOptions original =
        BuildOptions.builder()
            .addStarlarkOption(Label.parseCanonicalUnchecked("//custom:flag1"), "hello")
            .build();

    OptionsParser parser = OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build();
    parser.setStarlarkOptions(ImmutableMap.of("//custom:flag2", "foo"));

    assertThat(original.matches(parser)).isFalse();
  }

  @Test
  public void parsingResultMatchNullOption() throws Exception {
    BuildOptions original = BuildOptions.of(BUILD_CONFIG_OPTIONS);

    OptionsParser parser = OptionsParser.builder().optionsClasses(BUILD_CONFIG_OPTIONS).build();
    parser.parse("--null_option=foo"); // Note: null_option is null by default.

    assertThat(original.matches(parser)).isFalse();
  }
}
