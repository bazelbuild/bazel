// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.common.options.OptionsParser.newOptionsParser;
import static java.util.Arrays.asList;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.OptionValueDescription.SingleOptionValueDescription;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Predicate;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link OptionsParser}.
 */
@RunWith(JUnit4.class)
public class OptionsParserTest {

  /** Dummy comment (linter suppression) */
  public static class BadOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean foo1;

    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean foo2;
  }

  @Test
  public void errorsDuringConstructionAreWrapped() {
    try {
      newOptionsParser(BadOptions.class);
      fail();
    } catch (OptionsParser.ConstructionException e) {
      assertThat(e).hasCauseThat().isInstanceOf(DuplicateOptionDeclarationException.class);
    }
  }

  public static class ExampleFoo extends OptionsBase {

    @Option(
      name = "foo",
      category = "one",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "defaultFoo"
    )
    public String foo;

    @Option(
      name = "bar",
      category = "two",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "42"
    )
    public int bar;

    @Option(
      name = "bing",
      category = "one",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "",
      allowMultiple = true
    )
    public List<String> bing;

    @Option(
      name = "bang",
      category = "one",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "",
      converter = StringConverter.class,
      allowMultiple = true
    )
    public List<String> bang;

    @Option(
      name = "nodoc",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "",
      allowMultiple = false
    )
    public String nodoc;
  }

  public static class ExampleBaz extends OptionsBase {

    @Option(
      name = "baz",
      category = "one",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "defaultBaz"
    )
    public String baz;
  }

  /** Subclass of an options class. */
  public static class ExampleBazSubclass extends ExampleBaz {

    @Option(
      name = "baz_subclass",
      category = "one",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "defaultBazSubclass"
    )
    public String bazSubclass;
  }

  /**
   * Example with empty to null string converter
   */
  public static class ExampleBoom extends OptionsBase {
    @Option(
      name = "boom",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "defaultBoom",
      converter = EmptyToNullStringConverter.class
    )
    public String boom;
  }

  /**
   * Example with internal options
   */
  public static class ExampleInternalOptions extends OptionsBase {
    @Option(
      name = "internal_boolean",
      metadataTags = {OptionMetadataTag.INTERNAL},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "true"
    )
    public boolean privateBoolean;

    @Option(
      name = "internal_string",
      metadataTags = {OptionMetadataTag.INTERNAL},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "super secret"
    )
    public String privateString;
  }

  public static class StringConverter implements Converter<String> {
    @Override
    public String convert(String input) {
      return input;
    }
    @Override
    public String getTypeDescription() {
      return "a string";
    }
  }

  /**
   * A converter that defaults to null if the input is the empty string
   */
  public static class EmptyToNullStringConverter extends StringConverter {
    @Override
    public String convert(String input) {
      return input.isEmpty() ? null : input;
    }
  }

  @Test
  public void parseWithMultipleOptionsInterfaces()
      throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.parse("--baz=oops", "--bar", "17");
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("defaultFoo");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("oops");
  }

  @Test
  public void parseWithParamsFile() throws OptionsParsingException, IOException {
    // TODO(bazel-team): Switch to an in memory file system, here and below.
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--baz=oops --bar 17"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("defaultFoo");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("oops");
  }

  @Test
  public void parseWithEmptyParamsFile() throws OptionsParsingException, IOException {
    // TODO(bazel-team): Switch to an in memory file system, here and below.
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of(""),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("defaultFoo");
    assertThat(foo.bar).isEqualTo(42);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("defaultBaz");
  }

  @Test
  public void parseWithParamsFileWithEmptyStringValues() throws Exception {
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--baz", "", "--foo", ""),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEmpty();
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEmpty();
  }

  @Test
  public void parseWithParamsFileWithEmptyString() throws OptionsParsingException, IOException {
    // TODO(bazel-team): Switch to an in memory file system, here and below.
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--baz  --bar 17"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("defaultFoo");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEmpty();
  }

  @Test
  public void parseWithParamsFileWithEmptyStringAtEnd()
      throws OptionsParsingException, IOException {
    // TODO(bazel-team): Switch to an in memory file system, here and below.
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--bar",
            "17",
            " --baz",
            ""),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("defaultFoo");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEmpty();
  }

  @Test
  public void parseWithParamsFileWithQuotedSpaces() throws OptionsParsingException, IOException {
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--foo=\"fuzzy\nfoo\" --bar 17"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("\"fuzzy\nfoo\"");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("defaultBaz");
  }

  @Test
  public void parseWithParamsFileWithEscapedSpaces() throws OptionsParsingException, IOException {
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--foo=fuzzy\\ foo --bar 17"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("fuzzy\\ foo");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("defaultBaz");
  }

  @Test
  public void parseWithParamsFileWithEscapedQuotes() throws OptionsParsingException, IOException {
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--foo=\"fuzzy\\\"foo\" --bar 17"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("\"fuzzy\\\"foo\"");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("defaultBaz");
  }

  @Test
  public void parseWithParamsFileSingleQuotesUnescaping()
      throws OptionsParsingException, IOException {
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--foo", "'fuzzy '\\''foo'", "--bar", "17"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("fuzzy 'foo");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("defaultBaz");
  }

  @Test
  public void parseWithParamsFilePartiallyQuotedNoUnescaping()
      throws OptionsParsingException, IOException {
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--foo", "'fuzzy 'foo", "--bar", "17"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("'fuzzy 'foo");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("defaultBaz");
  }

  @Test
  public void parseWithParamsFileUnmatchedQuote() throws IOException {
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--foo=\"fuzzy foo --bar 17"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    try {
      parser.parse("@" + params);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo(
              String.format(
                  ParamsFilePreProcessor.ERROR_MESSAGE_FORMAT,
                  params,
                  String.format(ParamsFilePreProcessor.UNFINISHED_QUOTE_MESSAGE_FORMAT, "\"", 6)));
    }
  }

  @Test
  public void parseWithParamsFileWithMultilineStringValues() throws Exception {
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of(
            "--baz",
            "'hello\nworld'",
            "--foo",
            "hello\\",
            "world",
            "--nodoc",
            "\"hello",
            "world\""),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("hello\\\nworld");
    assertThat(foo.nodoc).isEqualTo("\"hello\nworld\"");
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("hello\nworld");
  }

  @Test
  public void parseWithParamsFileWithMultilineStringValuesCRLF() throws Exception {
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of(
            "--baz\r\n'hello\nworld'\r\n--foo\r\nhello\\\r\nworld\r\n\r\n"
            + "--nodoc\r\n\"hello\r\nworld\""),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("hello\nworld");
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("hello\\\nworld");
    assertThat(foo.nodoc).isEqualTo("\"hello\nworld\"");
  }

  @Test
  public void parseWithParamsFileMultiline() throws OptionsParsingException, IOException {
    // TODO(bazel-team): Switch to an in memory file system.
    Path params = Files.createTempDirectory("foo").resolve("params");
    Files.write(
        params,
        ImmutableList.of("--baz", "oops", "--bar", "17"),
        StandardCharsets.UTF_8,
        StandardOpenOption.CREATE);

    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    parser.parse("@" + params);
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("defaultFoo");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("oops");
  }

  @Test
  public void parsingFailsWithMissingParamsFile() {
    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.enableParamsFileSupport(FileSystems.getDefault());
    List<String> unknownOpts = asList("@does/not/exist");
    try {
      parser.parse(unknownOpts);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e.getInvalidArgument()).isEqualTo("@does/not/exist");
      assertThat(parser.getOptions(ExampleFoo.class)).isNotNull();
      assertThat(parser.getOptions(ExampleBaz.class)).isNotNull();
    }
  }

  @Test
  public void parseWithOptionsInheritance() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(ExampleBazSubclass.class);
    parser.parse("--baz_subclass=cat", "--baz=dog");
    ExampleBazSubclass subclassOptions = parser.getOptions(ExampleBazSubclass.class);
    assertThat(subclassOptions.bazSubclass).isEqualTo("cat");
    assertThat(subclassOptions.baz).isEqualTo("dog");
    ExampleBaz options = parser.getOptions(ExampleBaz.class);
    // This is a test showcasing the lack of functionality for retrieving parsed options at a
    // superclass type class type. If there's a need for this functionality, we can add it later.
    assertThat(options).isNull();
  }

  @Test
  public void parserWithUnknownOption() {
    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    try {
      parser.parse("--unknown", "option");
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e.getInvalidArgument()).isEqualTo("--unknown");
      assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: --unknown");
    }
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void parserWithSingleDashOption() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    try {
      parser.parse("-baz=oops", "-bar", "17");
      fail();
    } catch (OptionsParsingException expected) {}

    parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.setAllowSingleDashLongOptions(true);
    parser.parse("-baz=oops", "-bar", "17");
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("defaultFoo");
    assertThat(foo.bar).isEqualTo(17);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(baz.baz).isEqualTo("oops");
  }

  @Test
  public void parsingFailsWithUnknownOptions() {
    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    List<String> unknownOpts = asList("--unknown", "option", "--more_unknowns");
    try {
      parser.parse(unknownOpts);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e.getInvalidArgument()).isEqualTo("--unknown");
      assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: --unknown");
      assertThat(parser.getOptions(ExampleFoo.class)).isNotNull();
      assertThat(parser.getOptions(ExampleBaz.class)).isNotNull();
    }
  }

  @Test
  public void parsingFailsWithInternalBooleanOptionAsIfUnknown() {
    OptionsParser parser = newOptionsParser(ExampleInternalOptions.class);
    List<String> internalOpts = asList("--internal_boolean");
    try {
      parser.parse(internalOpts);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e.getInvalidArgument()).isEqualTo("--internal_boolean");
      assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: --internal_boolean");
      assertThat(parser.getOptions(ExampleInternalOptions.class)).isNotNull();
    }
  }

  @Test
  public void parsingFailsWithNegatedInternalBooleanOptionAsIfUnknown() {
    OptionsParser parser = newOptionsParser(ExampleInternalOptions.class);
    List<String> internalOpts = asList("--nointernal_boolean");
    try {
      parser.parse(internalOpts);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e.getInvalidArgument()).isEqualTo("--nointernal_boolean");
      assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: --nointernal_boolean");
      assertThat(parser.getOptions(ExampleInternalOptions.class)).isNotNull();
    }
  }

  @Test
  public void parsingFailsForInternalOptionWithValueInSameArgAsIfUnknown() {
    OptionsParser parser = newOptionsParser(ExampleInternalOptions.class);
    List<String> internalOpts = asList("--internal_string=any_value");
    try {
      parser.parse(internalOpts);
      fail("parsing should have failed for including a private option");
    } catch (OptionsParsingException e) {
      assertThat(e.getInvalidArgument()).isEqualTo("--internal_string=any_value");
      assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: --internal_string=any_value");
      assertThat(parser.getOptions(ExampleInternalOptions.class)).isNotNull();
    }
  }

  @Test
  public void parsingFailsForInternalOptionWithValueInSeparateArgAsIfUnknown() {
    OptionsParser parser = newOptionsParser(ExampleInternalOptions.class);
    List<String> internalOpts = asList("--internal_string", "any_value");
    try {
      parser.parse(internalOpts);
      fail("parsing should have failed for including a private option");
    } catch (OptionsParsingException e) {
      assertThat(e.getInvalidArgument()).isEqualTo("--internal_string");
      assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: --internal_string");
      assertThat(parser.getOptions(ExampleInternalOptions.class)).isNotNull();
    }
  }

  @Test
  public void parseKnownAndUnknownOptions() {
    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    List<String> opts = asList("--bar", "17", "--unknown", "option");
    try {
      parser.parse(opts);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e.getInvalidArgument()).isEqualTo("--unknown");
      assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: --unknown");
      assertThat(parser.getOptions(ExampleFoo.class)).isNotNull();
      assertThat(parser.getOptions(ExampleBaz.class)).isNotNull();
    }
  }

  @Test
  public void parseAndOverrideWithEmptyStringToObtainNullValueInOption()
      throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(ExampleBoom.class);
    // Override --boom value to the empty string
    parser.parse("--boom=");
    ExampleBoom boom = parser.getOptions(ExampleBoom.class);
    // The converted value is intentionally null since boom uses the EmptyToNullStringConverter
    assertThat(boom.boom).isNull();
  }

  public static class CategoryTest extends OptionsBase {
    @Option(
      name = "swiss_bank_account_number",
      documentationCategory =
          OptionDocumentationCategory.UNDOCUMENTED, // Not printed in usage messages!
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "123456789"
    )
    public int swissBankAccountNumber;

    @Option(
      name = "student_bank_account_number",
      category = "one",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "987654321"
    )
    public int studentBankAccountNumber;
  }

  @Test
  public void getOptionsAndGetResidueWithNoCallToParse() {
    // With no call to parse(), all options are at default values, and there's
    // no reside.
    assertThat(newOptionsParser(ExampleFoo.class).getOptions(ExampleFoo.class).foo)
        .isEqualTo("defaultFoo");
    assertThat(newOptionsParser(ExampleFoo.class).getResidue()).isEmpty();
  }

  @Test
  public void parserCanBeCalledRepeatedly() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(ExampleFoo.class);
    parser.parse("--foo", "foo1");
    assertThat(parser.getOptions(ExampleFoo.class).foo).isEqualTo("foo1");
    parser.parse();
    assertThat(parser.getOptions(ExampleFoo.class).foo).isEqualTo("foo1"); // no change
    parser.parse("--foo", "foo2");
    assertThat(parser.getOptions(ExampleFoo.class).foo).isEqualTo("foo2"); // updated
  }

  @Test
  public void multipleOccuringOption() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(ExampleFoo.class);
    parser.parse("--bing", "abcdef", "--foo", "foo1", "--bing", "123456" );
    assertThat(parser.getOptions(ExampleFoo.class).bing).containsExactly("abcdef", "123456");
  }

  @Test
  public void multipleOccurringOptionWithConverter() throws OptionsParsingException {
    // --bang is the same as --bing except that it has a "converter" specified.
    // This test also tests option values with embedded commas and spaces.
    OptionsParser parser = newOptionsParser(ExampleFoo.class);
    parser.parse("--bang", "abc,def ghi", "--foo", "foo1", "--bang", "123456" );
    assertThat(parser.getOptions(ExampleFoo.class).bang).containsExactly("abc,def ghi", "123456");
  }

  @Test
  public void parserIgnoresOptionsAfterMinusMinus()
      throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.parse("--foo", "well", "--baz", "here", "--", "--bar", "ignore");
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    ExampleBaz baz = parser.getOptions(ExampleBaz.class);
    assertThat(foo.foo).isEqualTo("well");
    assertThat(baz.baz).isEqualTo("here");
    assertThat(foo.bar).isEqualTo(42); // the default!
    assertThat(parser.getResidue()).containsExactly("--bar", "ignore").inOrder();
  }

  @Test
  public void parserThrowsExceptionIfResidueIsNotAllowed() {
    OptionsParser parser = newOptionsParser(ExampleFoo.class);
    parser.setAllowResidue(false);
    try {
      parser.parse("residue", "is", "not", "OK");
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e).hasMessageThat().isEqualTo("Unrecognized arguments: residue is not OK");
    }
  }

  @Test
  public void multipleCallsToParse() throws Exception {
    OptionsParser parser = newOptionsParser(ExampleFoo.class);
    parser.setAllowResidue(true);
    parser.parse("--foo", "one", "--bar", "43", "unknown1");
    parser.parse("--foo", "two", "unknown2");
    ExampleFoo foo = parser.getOptions(ExampleFoo.class);
    assertThat(foo.foo).isEqualTo("two"); // second call takes precedence
    assertThat(foo.bar).isEqualTo(43);
    assertThat(parser.getResidue()).containsExactly("unknown1", "unknown2").inOrder();
  }

  // Regression test for a subtle bug!  The toString of each options interface
  // instance was printing out key=value pairs for all flags in the
  // OptionsParser, not just those belonging to the specific interface type.
  @Test
  public void toStringDoesntIncludeFlagsForOtherOptionsInParserInstance()
      throws Exception {
    OptionsParser parser = newOptionsParser(ExampleFoo.class, ExampleBaz.class);
    parser.parse("--foo", "foo", "--bar", "43", "--baz", "baz");

    String fooString = parser.getOptions(ExampleFoo.class).toString();
    if (!fooString.contains("foo=foo") ||
        !fooString.contains("bar=43") ||
        !fooString.contains("ExampleFoo") ||
        fooString.contains("baz=baz")) {
      fail("ExampleFoo.toString() is incorrect: " + fooString);
    }

    String bazString = parser.getOptions(ExampleBaz.class).toString();
    if (!bazString.contains("baz=baz") ||
        !bazString.contains("ExampleBaz") ||
        bazString.contains("foo=foo") ||
        bazString.contains("bar=43")) {
      fail("ExampleBaz.toString() is incorrect: " + bazString);
    }
  }

  // Regression test for another subtle bug!  The toString was printing all the
  // explicitly-specified options, even if they were at their default values,
  // causing toString equivalence to diverge from equals().
  @Test
  public void toStringIsIndependentOfExplicitCommandLineOptions() throws Exception {
    ExampleFoo foo1 = Options.parse(ExampleFoo.class).getOptions();
    ExampleFoo foo2 = Options.parse(ExampleFoo.class, "--bar", "42").getOptions();
    assertThat(foo2).isEqualTo(foo1);
    assertThat(foo2.toString()).isEqualTo(foo1.toString());

    Map<String, Object> expectedMap = new ImmutableMap.Builder<String, Object>().
        put("bing", Collections.emptyList()).
        put("bar", 42).
        put("nodoc", "").
        put("bang", Collections.emptyList()).
        put("foo", "defaultFoo").build();

    assertThat(foo1.asMap()).isEqualTo(expectedMap);
    assertThat(foo2.asMap()).isEqualTo(expectedMap);
  }

  // Regression test for yet another subtle bug!  The inherited options weren't
  // being printed by toString.  One day, a real rain will come and wash all
  // this scummy code off the streets.
  public static class DerivedBaz extends ExampleBaz {
    @Option(
      name = "derived",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "defaultDerived"
    )
    public String derived;
  }

  @Test
  public void toStringPrintsInheritedOptionsToo_Duh() throws Exception {
    DerivedBaz derivedBaz = Options.parse(DerivedBaz.class).getOptions();
    String derivedBazString = derivedBaz.toString();
    if (!derivedBazString.contains("derived=defaultDerived") ||
        !derivedBazString.contains("baz=defaultBaz")) {
      fail("DerivedBaz.toString() is incorrect: " + derivedBazString);
    }
  }

  // Tests for new default value override mechanism
  public static class CustomOptions extends OptionsBase {
    @Option(
      name = "simple",
      category = "custom",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "simple default"
    )
    public String simple;

    @Option(
      name = "multipart_name",
      category = "custom",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "multipart default"
    )
    public String multipartName;
  }

  @Test
  public void assertDefaultStringsForCustomOptions() throws OptionsParsingException {
    CustomOptions options = Options.parse(CustomOptions.class).getOptions();
    assertThat(options.simple).isEqualTo("simple default");
    assertThat(options.multipartName).isEqualTo("multipart default");
  }

  public static class NullTestOptions extends OptionsBase {
    @Option(
      name = "simple",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String simple;
  }

  @Test
  public void defaultNullStringGivesNull() throws Exception {
    NullTestOptions options = Options.parse(NullTestOptions.class).getOptions();
    assertThat(options.simple).isNull();
  }

  public static class ImplicitDependencyOptions extends OptionsBase {
    @Option(
      name = "first",
      implicitRequirements = "--second=second",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String first;

    @Option(
      name = "second",
      implicitRequirements = "--third=third",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String second;

    @Option(
      name = "third",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String third;
  }

  @Test
  public void implicitDependencyHasImplicitDependency() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ImplicitDependencyOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--first=first"));
    assertThat(parser.getOptions(ImplicitDependencyOptions.class).first).isEqualTo("first");
    assertThat(parser.getOptions(ImplicitDependencyOptions.class).second).isEqualTo("second");
    assertThat(parser.getOptions(ImplicitDependencyOptions.class).third).isEqualTo("third");
  }

  public static class BadImplicitDependencyOptions extends OptionsBase {
    @Option(
      name = "first",
      implicitRequirements = "xxx",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String first;
  }

  @Test
  public void badImplicitDependency() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(BadImplicitDependencyOptions.class);
    try {
      parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--first=first"));
    } catch (AssertionError e) {
      /* Expected error. */
      return;
    }
    fail();
  }

  public static class BadExpansionOptions extends OptionsBase {
    @Option(
      name = "first",
      expansion = {"xxx"},
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public Void first;
  }

  @Test
  public void badExpansionOptions() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(BadExpansionOptions.class);
    try {
      parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--first"));
    } catch (AssertionError e) {
      /* Expected error. */
      return;
    }
    fail();
  }

  /** NullExpansionOptions */
  public static class NullExpansionsOptions extends OptionsBase {

    /** ExpFunc */
    public static class ExpFunc implements ExpansionFunction {
      @Override
      public ImmutableList<String> getExpansion(ExpansionContext context) {
        return null;
      }
    }

    @Option(
      name = "badness",
      expansionFunction = ExpFunc.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public Void badness;
  }

  @Test
  public void nullExpansions() throws Exception {
    // Ensure that we get the NPE at the time of parser construction, not later when actually
    // parsing.
    try {
      newOptionsParser(NullExpansionsOptions.class);
      fail("Should have failed due to null expansion function result");
    } catch (OptionsParser.ConstructionException e) {
      assertThat(e).hasCauseThat().isInstanceOf(IllegalStateException.class);
    }
  }

  /** NullExpansionOptions */
  public static class NullExpansionsWithArgumentOptions extends OptionsBase {

    /** ExpFunc */
    public static class ExpFunc implements ExpansionFunction {
      @Override
      public ImmutableList<String> getExpansion(ExpansionContext context) {
        return null;
      }
    }

    @Option(
      name = "badness",
      expansionFunction = ExpFunc.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String badness;
  }

  @Test
  public void nullExpansionsWithArgument() throws Exception {
    try {
      // When an expansion takes a value, this exception should still happen at parse time.
      newOptionsParser(NullExpansionsWithArgumentOptions.class);
      fail("Should have failed due to null expansion function result");
    } catch (OptionsParser.ConstructionException e) {
      assertThat(e)
          .hasMessageThat()
          .isEqualTo("Error calling expansion function for option: badness");
    }
  }

  /** ExpansionOptions */
  public static class ExpansionOptions extends OptionsBase {
    @Option(
      name = "underlying",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String underlying;

    @Option(
      name = "expands",
      expansion = {"--underlying=from_expansion"},
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public Void expands;

    /** ExpFunc */
    public static class ExpFunc implements ExpansionFunction {
      @Override
      public ImmutableList<String> getExpansion(ExpansionContext context) {
        return ImmutableList.of("--expands");
      }
    }

    @Option(
      name = "expands_by_function",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      expansionFunction = ExpFunc.class
    )
    public Void expandsByFunction;
  }

  /** ExpansionMultipleOptions */
  public static class ExpansionMultipleOptions extends OptionsBase {
    @Option(
      name = "underlying",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      allowMultiple = true
    )
    public List<String> underlying;

    /** ExpFunc */
    public static class ExpFunc implements ExpansionFunction {
      @Override
      public ImmutableList<String> getExpansion(ExpansionContext context)
          throws OptionsParsingException {
        String value = context.getUnparsedValue();
        if (value == null) {
          throw new ExpansionNeedsValueException("No value given to 'expands_by_function'");
        }

        return ImmutableList.of("--underlying=pre_" + value, "--underlying=post_" + value);
      }
    }

    @Option(
      name = "expands_by_function",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      expansionFunction = ExpFunc.class
    )
    public Void expandsByFunction;
  }

  @Test
  public void describeOptionsWithExpansion() throws Exception {
    // We have to test this here rather than in OptionsTest because expansion functions require
    // that an options parser be constructed.
    OptionsParser parser = OptionsParser.newOptionsParser(ExpansionOptions.class);
    String usage =
        parser.describeOptions(ImmutableMap.<String, String>of(), OptionsParser.HelpVerbosity.LONG);
    assertThat(usage).contains("  --expands\n      Expands to: --underlying=from_expansion");
    assertThat(usage).contains("  --expands_by_function\n      Expands to: --expands");
  }

  @Test
  public void overrideExpansionWithExplicit() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ExpansionOptions.class);
    parser.parse(
        OptionPriority.COMMAND_LINE, null, Arrays.asList("--expands", "--underlying=direct_value"));
    ExpansionOptions options = parser.getOptions(ExpansionOptions.class);
    assertThat(options.underlying).isEqualTo("direct_value");
    assertThat(parser.getWarnings()).isEmpty();
  }

  @Test
  public void overrideExplicitWithExpansion() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ExpansionOptions.class);
    parser.parse(
        OptionPriority.COMMAND_LINE, null, Arrays.asList("--underlying=direct_value", "--expands"));
    ExpansionOptions options = parser.getOptions(ExpansionOptions.class);
    assertThat(options.underlying).isEqualTo("from_expansion");
  }

  // Makes sure the expansion options are expanded in the right order if they affect flags that
  // allow multiples.
  @Test
  public void multipleExpansionOptionsWithValue() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ExpansionMultipleOptions.class);
    parser.parse(
        OptionPriority.COMMAND_LINE,
        null,
        Arrays.asList("--expands_by_function=a", "--expands_by_function=b"));
    ExpansionMultipleOptions options = parser.getOptions(ExpansionMultipleOptions.class);
    assertThat(options.underlying).containsExactly("pre_a", "post_a", "pre_b", "post_b").inOrder();
  }

  @Test
  public void overrideWithHigherPriority() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(NullTestOptions.class);
    parser.parse(OptionPriority.RC_FILE, null, Arrays.asList("--simple=a"));
    assertThat(parser.getOptions(NullTestOptions.class).simple).isEqualTo("a");
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--simple=b"));
    assertThat(parser.getOptions(NullTestOptions.class).simple).isEqualTo("b");
  }

  @Test
  public void overrideWithLowerPriority() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(NullTestOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--simple=a"));
    assertThat(parser.getOptions(NullTestOptions.class).simple).isEqualTo("a");
    parser.parse(OptionPriority.RC_FILE, null, Arrays.asList("--simple=b"));
    assertThat(parser.getOptions(NullTestOptions.class).simple).isEqualTo("a");
  }

  @Test
  public void getOptionValueDescriptionWithNonExistingOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(NullTestOptions.class);
    try {
      parser.getOptionValueDescription("notexisting");
      fail();
    } catch (IllegalArgumentException e) {
      /* Expected exception. */
    }
  }

  @Test
  public void getOptionValueDescriptionWithoutValue() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(NullTestOptions.class);
    assertThat(parser.getOptionValueDescription("simple")).isNull();
  }

  @Test
  public void getOptionValueDescriptionWithValue() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(NullTestOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, "my description",
        Arrays.asList("--simple=abc"));
    OptionValueDescription result = parser.getOptionValueDescription("simple");
    assertThat(result).isNotNull();
    assertThat(result.getOptionDefinition().getOptionName()).isEqualTo("simple");
    assertThat(result.getValue()).isEqualTo("abc");
    assertThat(result.getSourceString()).isEqualTo("my description");

    // To check that the option tracks origin correctly, we need to check information that is
    // specific to a single-valued option.
    SingleOptionValueDescription singleOptionResult = (SingleOptionValueDescription) result;
    ParsedOptionDescription singleOptionInstance = singleOptionResult.getEffectiveOptionInstance();
    assertThat(singleOptionInstance.getPriority()).isEqualTo(OptionPriority.COMMAND_LINE);
    assertThat(singleOptionInstance.getOptionDefinition().isExpansionOption()).isFalse();
  }

  public static class ImplicitDependencyWarningOptions extends OptionsBase {
    @Option(
      name = "first",
      implicitRequirements = "--second=second",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String first;

    @Option(
      name = "second",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String second;

    @Option(
      name = "third",
      implicitRequirements = "--second=third",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String third;
  }

  @Test
  public void warningForImplicitOverridingExplicitOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ImplicitDependencyWarningOptions.class);
    parser.parse("--second=second", "--first=first");
    assertThat(parser.getWarnings())
        .containsExactly("Option 'second' is implicitly defined by "
                         + "option 'first'; the implicitly set value overrides the previous one");
  }

  @Test
  public void warningForExplicitOverridingImplicitOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ImplicitDependencyWarningOptions.class);
    parser.parse("--first=first");
    assertThat(parser.getWarnings()).isEmpty();
    parser.parse("--second=second");
    assertThat(parser.getWarnings())
        .containsExactly("A new value for option 'second' overrides a"
                         + " previous implicit setting of that option by option 'first'");
  }

  @Test
  public void warningForExplicitOverridingImplicitOptionInSameCall() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ImplicitDependencyWarningOptions.class);
    parser.parse("--first=first", "--second=second");
    assertThat(parser.getWarnings())
        .containsExactly("Option 'second' is implicitly defined by "
                         + "option 'first'; the implicitly set value overrides the previous one");
  }

  @Test
  public void warningForImplicitOverridingImplicitOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ImplicitDependencyWarningOptions.class);
    parser.parse("--first=first");
    assertThat(parser.getWarnings()).isEmpty();
    parser.parse("--third=third");
    assertThat(parser.getWarnings())
        .containsExactly("Option 'second' is implicitly defined by both "
                         + "option 'first' and option 'third'");
  }

  public static class WarningOptions extends OptionsBase {
    @Deprecated
    @Option(
      name = "first",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public Void first;

    @Deprecated
    @Option(
      name = "second",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public List<String> second;

    @Deprecated
    @Option(
      name = "third",
      expansion = "--fourth=true",
      abbrev = 't',
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public Void third;

    @Option(
      name = "fourth",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean fourth;
  }

  @Test
  public void deprecationWarning() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(WarningOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--first"));
    assertThat(parser.getWarnings()).isEqualTo(Arrays.asList("Option 'first' is deprecated"));
  }

  @Test
  public void deprecationWarningForListOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(WarningOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--second=a"));
    assertThat(parser.getWarnings()).isEqualTo(Arrays.asList("Option 'second' is deprecated"));
  }

  @Test
  public void deprecationWarningForExpansionOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(WarningOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--third"));
    assertThat(parser.getWarnings()).isEqualTo(Arrays.asList("Option 'third' is deprecated"));
    assertThat(parser.getOptions(WarningOptions.class).fourth).isTrue();
  }

  @Test
  public void deprecationWarningForAbbreviatedExpansionOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(WarningOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("-t"));
    assertThat(parser.getWarnings()).isEqualTo(Arrays.asList("Option 'third' is deprecated"));
    assertThat(parser.getOptions(WarningOptions.class).fourth).isTrue();
  }

  public static class NewWarningOptions extends OptionsBase {
    @Option(
      name = "first",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      deprecationWarning = "it's gone"
    )
    public Void first;

    @Option(
      name = "second",
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      deprecationWarning = "sorry, no replacement"
    )
    public List<String> second;

    @Option(
      name = "third",
      expansion = "--fourth=true",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      deprecationWarning = "use --forth instead"
    )
    public Void third;

    @Option(
      name = "fourth",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean fourth;
  }

  @Test
  public void newDeprecationWarning() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(NewWarningOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--first"));
    assertThat(parser.getWarnings())
        .isEqualTo(Arrays.asList("Option 'first' is deprecated: it's gone"));
  }

  @Test
  public void newDeprecationWarningForListOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(NewWarningOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--second=a"));
    assertThat(parser.getWarnings())
        .isEqualTo(Arrays.asList("Option 'second' is deprecated: sorry, no replacement"));
  }

  @Test
  public void newDeprecationWarningForExpansionOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(NewWarningOptions.class);
    parser.parse(OptionPriority.COMMAND_LINE, null, Arrays.asList("--third"));
    assertThat(parser.getWarnings())
        .isEqualTo(Arrays.asList("Option 'third' is deprecated: use --forth instead"));
    assertThat(parser.getOptions(NewWarningOptions.class).fourth).isTrue();
  }

  public static class ExpansionWarningOptions extends OptionsBase {
    @Option(
      name = "first",
      expansion = "--underlying=expandedFromFirst",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public Void first;

    @Option(
      name = "second",
      expansion = "--underlying=expandedFromSecond",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public Void second;

    @Option(
      name = "underlying",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public String underlying;
  }

  @Test
  public void warningForExpansionOverridingExplicitOption() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ExpansionWarningOptions.class);
    parser.parse("--underlying=underlying", "--first");
    assertThat(parser.getWarnings()).containsExactly(
        "The option 'first' was expanded and now overrides a "
        + "previous explicitly specified option 'underlying'");
  }

  @Test
  public void warningForTwoConflictingExpansionOptions() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ExpansionWarningOptions.class);
    parser.parse("--first", "--second");
    assertThat(parser.getWarnings()).containsExactly(
        "The option 'underlying' was expanded to from both options 'first' " + "and 'second'");
  }

  // This test is here to make sure that nobody accidentally changes the
  // order of the enum values and breaks the implicit assumptions elsewhere
  // in the code.
  @Test
  public void optionPrioritiesAreCorrectlyOrdered() throws Exception {
    assertThat(OptionPriority.values()).hasLength(6);
    assertThat(OptionPriority.DEFAULT).isLessThan(OptionPriority.COMPUTED_DEFAULT);
    assertThat(OptionPriority.COMPUTED_DEFAULT).isLessThan(OptionPriority.RC_FILE);
    assertThat(OptionPriority.RC_FILE).isLessThan(OptionPriority.COMMAND_LINE);
    assertThat(OptionPriority.COMMAND_LINE).isLessThan(OptionPriority.INVOCATION_POLICY);
    assertThat(OptionPriority.INVOCATION_POLICY).isLessThan(OptionPriority.SOFTWARE_REQUIREMENT);
  }

  public static class IntrospectionExample extends OptionsBase {
    @Option(
      name = "alpha",
      category = "one",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "alphaDefaultValue"
    )
    public String alpha;

    @Option(
      name = "beta",
      category = "one",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "betaDefaultValue"
    )
    public String beta;

    @Option(
      name = "gamma",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "gammaDefaultValue"
    )
    public String gamma;

    @Option(
      name = "delta",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "deltaDefaultValue"
    )
    public String delta;

    @Option(
      name = "echo",
      metadataTags = {OptionMetadataTag.HIDDEN},
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "echoDefaultValue"
    )
    public String echo;
  }

  @Test
  public void asListOfUnparsedOptions() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(IntrospectionExample.class);
    parser.parse(OptionPriority.COMMAND_LINE, "source",
        Arrays.asList("--alpha=one", "--gamma=two", "--echo=three"));
    List<ParsedOptionDescription> result = parser.asCompleteListOfParsedOptions();
    assertThat(result).isNotNull();
    assertThat(result).hasSize(3);

    assertThat(result.get(0).getOptionDefinition().getOptionName()).isEqualTo("alpha");
    assertThat(result.get(0).isDocumented()).isTrue();
    assertThat(result.get(0).isHidden()).isFalse();
    assertThat(result.get(0).getUnconvertedValue()).isEqualTo("one");
    assertThat(result.get(0).getSource()).isEqualTo("source");
    assertThat(result.get(0).getPriority()).isEqualTo(OptionPriority.COMMAND_LINE);

    assertThat(result.get(1).getOptionDefinition().getOptionName()).isEqualTo("gamma");
    assertThat(result.get(1).isDocumented()).isFalse();
    assertThat(result.get(1).isHidden()).isFalse();
    assertThat(result.get(1).getUnconvertedValue()).isEqualTo("two");
    assertThat(result.get(1).getSource()).isEqualTo("source");
    assertThat(result.get(1).getPriority()).isEqualTo(OptionPriority.COMMAND_LINE);

    assertThat(result.get(2).getOptionDefinition().getOptionName()).isEqualTo("echo");
    assertThat(result.get(2).isDocumented()).isFalse();
    assertThat(result.get(2).isHidden()).isTrue();
    assertThat(result.get(2).getUnconvertedValue()).isEqualTo("three");
    assertThat(result.get(2).getSource()).isEqualTo("source");
    assertThat(result.get(2).getPriority()).isEqualTo(OptionPriority.COMMAND_LINE);
  }

  @Test
  public void asListOfExplicitOptions() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(IntrospectionExample.class);
    parser.parse(OptionPriority.COMMAND_LINE, "source",
        Arrays.asList("--alpha=one", "--gamma=two"));
    List<ParsedOptionDescription> result = parser.asListOfExplicitOptions();
    assertThat(result).isNotNull();
    assertThat(result).hasSize(2);

    assertThat(result.get(0).getOptionDefinition().getOptionName()).isEqualTo("alpha");
    assertThat(result.get(0).isDocumented()).isTrue();
    assertThat(result.get(0).getUnconvertedValue()).isEqualTo("one");
    assertThat(result.get(0).getSource()).isEqualTo("source");
    assertThat(result.get(0).getPriority()).isEqualTo(OptionPriority.COMMAND_LINE);

    assertThat(result.get(1).getOptionDefinition().getOptionName()).isEqualTo("gamma");
    assertThat(result.get(1).isDocumented()).isFalse();
    assertThat(result.get(1).getUnconvertedValue()).isEqualTo("two");
    assertThat(result.get(1).getSource()).isEqualTo("source");
    assertThat(result.get(1).getPriority()).isEqualTo(OptionPriority.COMMAND_LINE);
  }

  private void assertOptionValue(
      String expectedName, Object expectedValue, OptionValueDescription actual) {
    assertThat(actual).isNotNull();
    assertThat(actual.getOptionDefinition().getOptionName()).isEqualTo(expectedName);
    assertThat(actual.getValue()).isEqualTo(expectedValue);
  }

  private void assertOptionValue(
      String expectedName,
      Object expectedValue,
      OptionPriority expectedPriority,
      String expectedSource,
      SingleOptionValueDescription actual) {
    assertOptionValue(expectedName, expectedValue, actual);
    assertThat(actual.getSourceString()).isEqualTo(expectedSource);
    assertThat(actual.getEffectiveOptionInstance().getPriority()).isEqualTo(expectedPriority);
  }

  @Test
  public void asListOfEffectiveOptions() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(IntrospectionExample.class);
    parser.parse(OptionPriority.COMMAND_LINE, "command line source",
        Arrays.asList("--alpha=alphaValueSetOnCommandLine", "--gamma=gammaValueSetOnCommandLine"));
    List<OptionValueDescription> result = parser.asListOfEffectiveOptions();
    assertThat(result).isNotNull();
    assertThat(result).hasSize(5);
    HashMap<String,OptionValueDescription> map = new HashMap<String,OptionValueDescription>();
    for (OptionValueDescription description : result) {
      map.put(description.getOptionDefinition().getOptionName(), description);
    }

    // All options in IntrospectionExample are single-valued options, and so have a 1:1 relationship
    // with the --flag=value option instance they came from (if any).
    assertOptionValue(
        "alpha",
        "alphaValueSetOnCommandLine",
        OptionPriority.COMMAND_LINE,
        "command line source",
        (SingleOptionValueDescription) map.get("alpha"));
    assertOptionValue(
        "gamma",
        "gammaValueSetOnCommandLine",
        OptionPriority.COMMAND_LINE,
        "command line source",
        (SingleOptionValueDescription) map.get("gamma"));
    assertOptionValue("beta", "betaDefaultValue", map.get("beta"));
    assertOptionValue("delta", "deltaDefaultValue", map.get("delta"));
    assertOptionValue("echo", "echoDefaultValue", map.get("echo"));
  }

  // Regression tests for bug:
  // "--option from blazerc unexpectedly overrides --option from command line"
  public static class ListExample extends OptionsBase {
    @Option(
      name = "alpha",
      converter = StringConverter.class,
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public List<String> alpha;
  }

  @Test
  public void overrideListOptions() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(ListExample.class);
    parser.parse(OptionPriority.COMMAND_LINE, "a", Arrays.asList("--alpha=two"));
    parser.parse(OptionPriority.RC_FILE, "b", Arrays.asList("--alpha=one"));
    assertThat(parser.getOptions(ListExample.class).alpha).isEqualTo(Arrays.asList("one", "two"));
  }

  public static class CommaSeparatedOptionsExample extends OptionsBase {
    @Option(
      name = "alpha",
      converter = CommaSeparatedOptionListConverter.class,
      allowMultiple = true,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null"
    )
    public List<String> alpha;
  }

  @Test
  public void commaSeparatedOptionsWithAllowMultiple() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(CommaSeparatedOptionsExample.class);
    parser.parse(OptionPriority.COMMAND_LINE, "a", Arrays.asList("--alpha=one",
        "--alpha=two,three"));
    assertThat(parser.getOptions(CommaSeparatedOptionsExample.class).alpha)
        .isEqualTo(Arrays.asList("one", "two", "three"));
  }

  public static class Yesterday extends OptionsBase {

    @Option(
      name = "a",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "a"
    )
    public String a;

    @Option(
      name = "b",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "b"
    )
    public String b;

    @Option(
      name = "c",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      expansion = {"--a=0"}
    )
    public Void c;

    @Option(
      name = "d",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      allowMultiple = true
    )
    public List<String> d;

    @Option(
      name = "e",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      implicitRequirements = {"--a==1"}
    )
    public String e;

    @Option(
      name = "f",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      implicitRequirements = {"--b==1"}
    )
    public String f;

    @Option(
      name = "g",
      abbrev = 'h',
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean g;
  }

  public static List<String> canonicalize(Class<? extends OptionsBase> optionsClass, String... args)
      throws OptionsParsingException {

    OptionsParser parser = OptionsParser.newOptionsParser(
        ImmutableList.<Class<? extends OptionsBase>>of(optionsClass));
    parser.setAllowResidue(false);
    parser.parse(Arrays.asList(args));
    return parser.canonicalize();
  }

  @Test
  public void canonicalizeEasy() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--a=x")).containsExactly("--a=x");
  }

  @Test
  public void canonicalizeSkipDuplicate() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--a=y", "--a=x")).containsExactly("--a=x");
  }

  @Test
  public void canonicalizeExpands() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--c")).containsExactly("--a=0");
  }

  @Test
  public void canonicalizeExpansionOverridesExplicit() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--a=x", "--c")).containsExactly("--a=0");
  }

  @Test
  public void canonicalizeExplicitOverridesExpansion() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--c", "--a=x")).containsExactly("--a=x");
  }

  @Test
  public void canonicalizeSorts() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--b=y", "--a=x"))
        .containsExactly("--a=x", "--b=y").inOrder();
  }

  @Test
  public void canonicalizeImplicitDepsAtEnd() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--e=y", "--a=x"))
        .isEqualTo(Arrays.asList("--a=x", "--e=y"));
  }

  @Test
  public void canonicalizeImplicitDepsSkipsDuplicate() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--e=x", "--e=y")).containsExactly("--e=y");
  }

  @Test
  public void canonicalizeDoesNotSortImplicitDeps() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--f=z", "--e=y", "--a=x"))
        .containsExactly("--a=x", "--f=z", "--e=y").inOrder();
  }

  @Test
  public void canonicalizeDoesNotSkipAllowMultiple() throws Exception {
    assertThat(canonicalize(Yesterday.class, "--d=a", "--d=b"))
        .containsExactly("--d=a", "--d=b").inOrder();
  }

  @Test
  public void canonicalizeReplacesAbbrevWithName() throws Exception {
    assertThat(canonicalize(Yesterday.class, "-h")).containsExactly("--g=1");
  }

  public static class LongValueExample extends OptionsBase {
    @Option(
      name = "longval",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "2147483648"
    )
    public long longval;

    @Option(
      name = "intval",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "2147483647"
    )
    public int intval;
  }

  @Test
  public void parseLong() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(LongValueExample.class);
    parser.parse("");
    LongValueExample result = parser.getOptions(LongValueExample.class);
    assertThat(result.longval).isEqualTo(2147483648L);
    assertThat(result.intval).isEqualTo(2147483647);

    parser.parse("--longval", Long.toString(Long.MIN_VALUE));
    result = parser.getOptions(LongValueExample.class);
    assertThat(result.longval).isEqualTo(Long.MIN_VALUE);

    parser.parse("--longval", "100");
    result = parser.getOptions(LongValueExample.class);
    assertThat(result.longval).isEqualTo(100);
  }

  @Test
  public void intOutOfBounds() {
    OptionsParser parser = newOptionsParser(LongValueExample.class);
    try {
      parser.parse("--intval=2147483648");
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e).hasMessageThat().contains("'2147483648' is not an int");
    }
  }

  public static class OldNameExample extends OptionsBase {
    @Option(
      name = "new_name",
      oldName = "old_name",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "defaultValue"
    )
    public String flag;
  }

  @Test
  public void testOldName() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(OldNameExample.class);
    parser.parse("--old_name=foo");
    OldNameExample result = parser.getOptions(OldNameExample.class);
    assertThat(result.flag).isEqualTo("foo");

    // Should also work by its new name.
    parser = newOptionsParser(OldNameExample.class);
    parser.parse("--new_name=foo");
    result = parser.getOptions(OldNameExample.class);
    assertThat(result.flag).isEqualTo("foo");
    // Should be no warnings if the new name is used.
    assertThat(parser.getWarnings()).isEmpty();
  }

  @Test
  public void testOldNameCanonicalization() throws Exception {
    assertThat(canonicalize(OldNameExample.class, "--old_name=foo"))
        .containsExactly("--new_name=foo");
  }

  public static class ExampleBooleanFooOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean foo;
  }

  @Test
  public void testBooleanUnderscorePrefixError() {
    try {
      OptionsParser parser = newOptionsParser(ExampleBooleanFooOptions.class);
      parser.parse("--no_foo");

      fail("--no_foo should fail to parse.");
    } catch (OptionsParsingException e) {
      assertThat(e).hasMessageThat().contains("Unrecognized option: --no_foo");
    }
  }

  public static class WrapperOptionExample extends OptionsBase {
    @Option(
      name = "wrapper",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      wrapperOption = true
    )
    public Void wrapperOption;

    @Option(
      name = "flag1",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean flag1;

    @Option(
      name = "flag2",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "42"
    )
    public int flag2;

    @Option(
      name = "flag3",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "foo"
    )
    public String flag3;
  }

  @Test
  public void testWrapperOption() throws OptionsParsingException {
    OptionsParser parser = newOptionsParser(WrapperOptionExample.class);
    parser.parse("--wrapper=--flag1=true", "--wrapper=--flag2=87", "--wrapper=--flag3=bar");
    WrapperOptionExample result = parser.getOptions(WrapperOptionExample.class);
    assertThat(result.flag1).isTrue();
    assertThat(result.flag2).isEqualTo(87);
    assertThat(result.flag3).isEqualTo("bar");
  }

  @Test
  public void testInvalidWrapperOptionFormat() {
    OptionsParser parser = newOptionsParser(WrapperOptionExample.class);
    try {
      parser.parse("--wrapper=foo");
      fail();
    } catch (OptionsParsingException e) {
      // Check that the message looks like it's suggesting the correct format.
      assertThat(e).hasMessageThat().contains("--foo");
    }
  }

  @Test
  public void testWrapperCanonicalization() throws OptionsParsingException {
    List<String> canonicalized = canonicalize(WrapperOptionExample.class,
        "--wrapper=--flag1=true", "--wrapper=--flag2=87", "--wrapper=--flag3=bar");
    assertThat(canonicalized).isEqualTo(Arrays.asList("--flag1=true", "--flag2=87", "--flag3=bar"));
  }

  /** Dummy options that declares it uses only core types. */
  @UsesOnlyCoreTypes
  public static class CoreTypesOptions extends OptionsBase implements Serializable {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean foo;

    @Option(
      name = "bar",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "abc"
    )
    public String bar;
  }

  /** Dummy options that does not declare using only core types. */
  public static class NonCoreTypesOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean foo;
  }

  /** Dummy options that incorrectly claims to use only core types. */
  @UsesOnlyCoreTypes
  public static class BadCoreTypesOptions extends OptionsBase {
    /** Dummy unsafe type. */
    public static class Foo {
      public int i = 0;
    }

    /** Converter for Foo. */
    public static class FooConverter implements Converter<Foo> {
      @Override
      public Foo convert(String input) throws OptionsParsingException {
        Foo foo = new Foo();
        foo.i = Integer.parseInt(input);
        return foo;
      }

      @Override
      public String getTypeDescription() {
        return "a foo";
      }
    }

    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      converter = FooConverter.class
    )
    public Foo foo;
  }

  /** Dummy options that is unsafe for @UsesOnlyCoreTypes but doesn't use the annotation. */
  public static class SuperBadCoreTypesOptions extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      converter = BadCoreTypesOptions.FooConverter.class
    )
    public BadCoreTypesOptions.Foo foo;
  }

  /**
   * Dummy options that illegally advertises @UsesOnlyCoreTypes, when its direct fields are fine but
   * its inherited fields are not.
   */
  @UsesOnlyCoreTypes
  public static class InheritedBadCoreTypesOptions extends SuperBadCoreTypesOptions {
    @Option(
      name = "bar",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean bar;
  }

  @Test
  public void testUsesOnlyCoreTypes() {
    assertThat(OptionsParser.getUsesOnlyCoreTypes(CoreTypesOptions.class)).isTrue();
    assertThat(OptionsParser.getUsesOnlyCoreTypes(NonCoreTypesOptions.class)).isFalse();
  }

  @Test
  public void testValidationOfUsesOnlyCoreTypes() {
    try {
      OptionsParser.getUsesOnlyCoreTypes(BadCoreTypesOptions.class);
      fail("Should have detected illegal use of @UsesOnlyCoreTypes");
    } catch (OptionsParser.ConstructionException expected) {
      assertThat(expected)
          .hasMessageThat()
          .matches(
              "Options class '.*BadCoreTypesOptions' is marked as @UsesOnlyCoreTypes, but field "
                  + "'foo' has type '.*Foo'");
    }
  }

  @Test
  public void testValidationOfUsesOnlyCoreTypes_Inherited() {
    try {
      OptionsParser.getUsesOnlyCoreTypes(InheritedBadCoreTypesOptions.class);
      fail("Should have detected illegal use of @UsesOnlyCoreTypes "
          + "(due to inheritance from bad superclass)");
    } catch (OptionsParser.ConstructionException expected) {
      assertThat(expected)
          .hasMessageThat()
          .matches(
              "Options class '.*InheritedBadCoreTypesOptions' is marked as @UsesOnlyCoreTypes, but "
                  + "field 'foo' has type '.*Foo'");
    }
  }

  @Test
  public void serializable() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(CoreTypesOptions.class);
    parser.parse("--foo=true", "--bar=xyz");
    CoreTypesOptions options = parser.getOptions(CoreTypesOptions.class);

    ByteArrayOutputStream bos = new ByteArrayOutputStream();
    ObjectOutputStream objOut = new ObjectOutputStream(bos);
    objOut.writeObject(options);
    objOut.flush();
    ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
    ObjectInputStream objIn = new ObjectInputStream(bis);
    Object obj = objIn.readObject();

    assertThat(obj).isEqualTo(options);
  }

  @Test
  public void stableSerialization() throws Exception {
    // Construct options two different ways to get the same result, and confirm that the serialized
    // representation is identical.
    OptionsParser parser1 = OptionsParser.newOptionsParser(CoreTypesOptions.class);
    parser1.parse("--foo=true", "--bar=xyz");
    CoreTypesOptions options1 = parser1.getOptions(CoreTypesOptions.class);
    OptionsParser parser2 = OptionsParser.newOptionsParser(CoreTypesOptions.class);
    parser2.parse("--bar=abc", "--foo=1");
    CoreTypesOptions options2 = parser2.getOptions(CoreTypesOptions.class);
    options2.bar = "xyz";

    // We use two different pairs of streams because ObjectOutputStream#reset does not actually
    // wipe all the internal state. (The first time it's used, there's an additional header that
    // does not reappear afterwards.)
    ByteArrayOutputStream bos1 = new ByteArrayOutputStream();
    ObjectOutputStream objOut1 = new ObjectOutputStream(bos1);
    objOut1.writeObject(options1);
    objOut1.flush();
    byte[] data1 = bos1.toByteArray();
    ByteArrayOutputStream bos2 = new ByteArrayOutputStream();
    ObjectOutputStream objOut2 = new ObjectOutputStream(bos2);
    objOut2.writeObject(options2);
    objOut2.flush();
    byte[] data2 = bos2.toByteArray();

    assertThat(data1).isEqualTo(data2);
  }

  /** Dummy options for testing getHelpCompletion() and visitOptions(). */
  public static class CompletionOptions extends OptionsBase implements Serializable {
    @Option(
      name = "secret",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean secret;

    @Option(
        name = "b",
        documentationCategory = OptionDocumentationCategory.LOGGING,
        effectTags = {OptionEffectTag.NO_OP},
        defaultValue = "false"
      )
      public boolean b;

    @Option(
      name = "a",
      documentationCategory = OptionDocumentationCategory.QUERY,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "false"
    )
    public boolean a;
  }

  @Test
  public void getOptionsCompletionShouldFilterUndocumentedOptions() throws Exception {
    OptionsParser parser = OptionsParser.newOptionsParser(CompletionOptions.class);
    assertThat(parser.getOptionsCompletion().split("\n"))
        .isEqualTo(new String[] {"--a", "--noa", "--b", "--nob"});
  }

  @Test
  public void visitOptionsShouldFailWithoutPredicate() throws Exception {
    checkThatVisitOptionsThrowsNullPointerException(null, option -> {}, "Missing predicate.");
  }

  @Test
  public void visitOptionsShouldFailWithoutVisitor() throws Exception {
    checkThatVisitOptionsThrowsNullPointerException(option -> true, null, "Missing visitor.");
  }

  private void checkThatVisitOptionsThrowsNullPointerException(
      Predicate<OptionDefinition> predicate,
      Consumer<OptionDefinition> visitor,
      String expectedMessage)
      throws Exception {
    try {
      OptionsParser.newOptionsParser(CompletionOptions.class).visitOptions(predicate, visitor);
      fail("Expected a NullPointerException.");
    } catch (NullPointerException ex) {
      assertThat(ex).hasMessageThat().isEqualTo(expectedMessage);
    }
  }

  @Test
  public void visitOptionsShouldReturnAllOptionsInOrder() throws Exception {
    assertThat(visitOptionsToCollectTheirNames(option -> true)).containsExactly("a", "b", "secret");
  }

  @Test
  public void visitOptionsShouldObeyPredicate() throws Exception {
    assertThat(visitOptionsToCollectTheirNames(option -> false)).isEmpty();
    assertThat(visitOptionsToCollectTheirNames(option -> option.getOptionName().length() > 1))
        .containsExactly("secret");
  }

  private List<String> visitOptionsToCollectTheirNames(Predicate<OptionDefinition> predicate) {
    List<String> names = new LinkedList<>();
    Consumer<OptionDefinition> visitor = option -> names.add(option.getOptionName());

    OptionsParser parser = OptionsParser.newOptionsParser(CompletionOptions.class);
    parser.visitOptions(predicate, visitor);

    return names;
  }
}
