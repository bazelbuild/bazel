// Copyright 2017 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.fail;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import com.google.devtools.common.options.Converters.IntegerConverter;
import com.google.devtools.common.options.Converters.StringConverter;
import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link OptionDefinition}. */
@RunWith(JUnit4.class)
public class OptionDefinitionTest {

  /** Dummy options class, to test various expected failures of the OptionDefinition. */
  public static class BrokenOptions extends OptionsBase {
    @Option(
      name = "missing_its_converter",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "1"
    )
    public Map<String, String> noConverter;

    @Option(
      name = "multiple_but_not_a_list",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "null",
      allowMultiple = true
    )
    public String multipleWithNoList;

    @Option(
      name = "multiple_with_wrong_collection",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "1",
      allowMultiple = true
    )
    public Set<String> multipleWithSetType;

    @Option(
      name = "invalid_default",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "not a number"
    )
    public int invalidDefault;
  }

  @Test
  public void errorForMissingOptionConverter() throws Exception {
    OptionDefinition optionDef =
        OptionDefinition.extractOptionDefinition(BrokenOptions.class.getField("noConverter"));
    try {
      optionDef.getConverter();
      fail("Missing converter should have caused getConverter to fail.");
    } catch (ConstructionException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Option noConverter expects values of type java.util.Map<java.lang.String, "
                  + "java.lang.String>, but no converter was found; possible fix: "
                  + "add converter=... to its @Option annotation.");
    }
  }

  @Test
  public void errorForInvalidOptionTypeForRepeatableOptions() throws Exception {
    OptionDefinition optionDef =
        OptionDefinition.extractOptionDefinition(
            BrokenOptions.class.getField("multipleWithNoList"));
    try {
      optionDef.getConverter();
      fail("Mistyped allowMultiple option did not fail getConverter().");
    } catch (ConstructionException e) {
      assertThat(e)
          .hasMessageThat()
          .contains
              ("Option multipleWithNoList allows multiple occurrences, so must be of type "
                  + "List<...>");
    }
  }

  @Test
  public void errorForInvalidCollectionOptionConverter() throws Exception {
    OptionDefinition optionDef =
        OptionDefinition.extractOptionDefinition(
            BrokenOptions.class.getField("multipleWithSetType"));
    try {
      optionDef.getConverter();
      fail("Mistyped allowMultiple option did not fail getConverter().");
    } catch (ConstructionException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "Option multipleWithSetType allows multiple occurrences, so must be of type "
                  + "List<...>");
    }
  }

  @Test
  public void errorForInvalidDefaultValue() throws Exception {
    OptionDefinition optionDef =
        OptionDefinition.extractOptionDefinition(BrokenOptions.class.getField("invalidDefault"));
    try {
      optionDef.getDefaultValue();
      fail("Invalid default value parsed without failure.");
    } catch (ConstructionException e) {
      assertThat(e)
          .hasMessageThat()
          .contains(
              "OptionsParsingException while retrieving the default value for invalidDefault: "
                  + "'not a number' is not an int");
    }
  }

  /** The rare valid options. */
  public static class ValidOptionUsingDefaultConverterForMocking extends OptionsBase {
    @Option(
      name = "foo",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "42"
    )
    public int foo;

    @Option(
      name = "bar",
      converter = StringConverter.class,
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.NO_OP},
      defaultValue = "strings"
    )
    public int bar;
  }

  /**
   * Test that the converter and option default values are only computed once and then are obtained
   * from the stored values, in the case where a default converter is used.
   */
  @Test
  public void optionDefinitionMemoizesDefaultConverterValue() throws Exception {
    OptionDefinition optionDefinition =
        OptionDefinition.extractOptionDefinition(
            ValidOptionUsingDefaultConverterForMocking.class.getField("foo"));
    OptionDefinition mockOptionDef = Mockito.spy(optionDefinition);

    // Do a bunch of potentially repeat operations on this option that need to know information
    // about the converter and default value. Also verify that the values are as expected.
    boolean isBoolean = mockOptionDef.isBooleanField();
    assertThat(isBoolean).isFalse();

    Converter<?> converter = mockOptionDef.getConverter();
    assertThat(converter).isInstanceOf(IntegerConverter.class);

    int value = (int) mockOptionDef.getDefaultValue();
    assertThat(value).isEqualTo(42);

    // Expect reference equality, since we didn't recompute the value
    Converter<?> secondConverter = mockOptionDef.getConverter();
    assertThat(secondConverter).isSameAs(converter);

    mockOptionDef.getDefaultValue();

    // Verify that we didn't re-calculate the converter from the provided class object.
    verify(mockOptionDef, times(1)).getProvidedConverter();
    // The first call to getDefaultValue checks isSpecialNullDefault, which called
    // getUnparsedValueDefault as well, but expect no more calls to it after the initial call.
    verify(mockOptionDef, times(1)).isSpecialNullDefault();
    verify(mockOptionDef, times(2)).getUnparsedDefaultValue();
  }

  /**
   * Test that the converter and option default values are only computed once and then are obtained
   * from the stored values, in the case where a converter was provided.
   */
  @Test
  public void optionDefinitionMemoizesProvidedConverterValue() throws Exception {
    OptionDefinition optionDefinition =
        OptionDefinition.extractOptionDefinition(
            ValidOptionUsingDefaultConverterForMocking.class.getField("bar"));
    OptionDefinition mockOptionDef = Mockito.spy(optionDefinition);

    // Do a bunch of potentially repeat operations on this option that need to know information
    // about the converter and default value. Also verify that the values are as expected.
    boolean isBoolean = mockOptionDef.isBooleanField();
    assertThat(isBoolean).isFalse();

    Converter<?> converter = mockOptionDef.getConverter();
    assertThat(converter).isInstanceOf(StringConverter.class);

    String value = (String) mockOptionDef.getDefaultValue();
    assertThat(value).isEqualTo("strings");

    // Expect reference equality, since we didn't recompute the value
    Converter<?> secondConverter = mockOptionDef.getConverter();
    assertThat(secondConverter).isSameAs(converter);

    mockOptionDef.getDefaultValue();

    // Verify that we didn't re-calculate the converter from the provided class object.
    verify(mockOptionDef, times(1)).getProvidedConverter();
    // The first call to getDefaultValue checks isSpecialNullDefault, which called
    // getUnparsedValueDefault as well, but expect no more calls to it after the initial call.
    verify(mockOptionDef, times(1)).isSpecialNullDefault();
    verify(mockOptionDef, times(2)).getUnparsedDefaultValue();
  }
}
