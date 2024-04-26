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

package com.google.devtools.common.options.testing;

import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.testing.EqualsTester;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.ArrayList;
import java.util.LinkedHashSet;

/**
 * A tester to confirm that {@link Converter} instances produce equal results on multiple calls with
 * the same input.
 */
public final class ConverterTester {

  private final Converter<?> converter;
  private final Class<? extends Converter<?>> converterClass;
  private final Object conversionContext;
  private final EqualsTester tester = new EqualsTester();
  private final LinkedHashSet<String> testedInputs = new LinkedHashSet<>();
  private final ArrayList<ImmutableList<String>> inputLists = new ArrayList<>();

  /** Creates a new ConverterTester which will test the given Converter class. */
  public ConverterTester(Class<? extends Converter<?>> converterClass, Object conversionContext) {
    this.converterClass = converterClass;
    this.converter = createConverter();
    this.conversionContext = conversionContext;
  }

  private Converter<?> createConverter() {
    try {
      return converterClass.getDeclaredConstructor().newInstance();
    } catch (ReflectiveOperationException ex) {
      throw new AssertionError("Failed to create converter", ex);
    }
  }

  /** Returns the class this ConverterTester is testing. */
  public Class<? extends Converter<?>> getConverterClass() {
    return converterClass;
  }

  /**
   * Returns whether this ConverterTester has a test for the given input, i.e., addEqualityGroup
   * was called with the given string.
   */
  public boolean hasTestForInput(String input) {
    return testedInputs.contains(input);
  }

  /**
   * Adds a set of valid inputs which are expected to convert to equal values.
   *
   * <p>The inputs added here will be converted to values using the Converter class passed to the
   * constructor of this instance; the resulting values must be equal (and have equal hashCodes):
   *
   * <ul>
   *   <li>to themselves
   *   <li>to another copy of themselves generated from the same Converter instance
   *   <li>to another copy of themselves generated from a different Converter instance
   *   <li>to the other values converted from inputs in the same addEqualityGroup call
   * </ul>
   *
   * <p>They must NOT be equal:
   *
   * <ul>
   *   <li>to null
   *   <li>to an instance of an arbitrary class
   *   <li>to any values converted from inputs in a different addEqualityGroup call
   * </ul>
   *
   * @throws AssertionError if an {@link OptionsParsingException} is thrown from the {@link
   *     Converter#convert} method when converting any of the inputs.
   * @see EqualsTester#addEqualityGroup
   */
  @CanIgnoreReturnValue
  public ConverterTester addEqualityGroup(String... inputs) {
    ImmutableList.Builder<WrappedItem> wrapped = ImmutableList.builder();
    ImmutableList<String> inputList = ImmutableList.copyOf(inputs);
    inputLists.add(inputList);
    for (String input : inputList) {
      testedInputs.add(input);
      try {
        wrapped.add(new WrappedItem(input, converter.convert(input, conversionContext)));
      } catch (OptionsParsingException ex) {
        throw new AssertionError("Failed to parse input: \"" + input + "\"", ex);
      }
    }
    tester.addEqualityGroup(wrapped.build().toArray());
    return this;
  }

  /**
   * Tests the convert method of the wrapped Converter class, verifying the properties listed in the
   * Javadoc listed for {@link #addEqualityGroup}.
   *
   * @throws AssertionError if one of the expected properties did not hold up
   * @see EqualsTester#testEquals
   */
  @CanIgnoreReturnValue
  public ConverterTester testConvert() {
    tester.testEquals();
    testItems();
    return this;
  }

  private void testItems() {
    for (ImmutableList<String> inputList : inputLists) {
      for (String input : inputList) {
        Converter<?> converter = createConverter();
        Converter<?> converter2 = createConverter();

        Object converted;
        Object convertedAgain;
        Object convertedDifferentConverterInstance;
        try {
          converted = converter.convert(input, conversionContext);
          convertedAgain = converter.convert(input, conversionContext);
          convertedDifferentConverterInstance = converter2.convert(input, conversionContext);
        } catch (OptionsParsingException ex) {
          throw new AssertionError("Failed to parse input: \"" + input + "\"", ex);
        }

        assertWithMessage(
                "Input \""
                    + input
                    + "\" was not equal to itself when converted twice by the same Converter")
            .that(convertedAgain)
            .isEqualTo(converted);
        assertWithMessage(
                "Input \""
                    + input
                    + "\" did not have a consistent hashCode when converted twice "
                    + "by the same Converter")
            .that(convertedAgain.hashCode())
            .isEqualTo(converted.hashCode());
        assertWithMessage(
            "Input \""
                + input
                + "\" was not equal to itself when converted twice by a different Converter")
            .that(convertedDifferentConverterInstance)
            .isEqualTo(converted);
        assertWithMessage(
            "Input \""
                + input
                + "\" did not have a consistent hashCode when converted twice "
                + "by a different Converter")
            .that(convertedDifferentConverterInstance.hashCode())
            .isEqualTo(converted.hashCode());
      }
    }
  }

  /**
   * A wrapper around the objects passed to EqualsTester to give them a more useful toString() so
   * that the mapping between the input text which actually appears in the source file and the
   * object produced from parsing it is more obvious.
   */
  private static final class WrappedItem {
    private final String argument;
    private final Object wrapped;

    private WrappedItem(String argument, Object wrapped) {
      this.argument = argument;
      this.wrapped = wrapped;
    }

    @Override
    public String toString() {
      return String.format("Converted input \"%s\" => [%s]", argument, wrapped);
    }

    @Override
    public int hashCode() {
      return wrapped.hashCode();
    }

    @Override
    public boolean equals(Object other) {
      if (other instanceof WrappedItem wrappedItem) {
        return this.wrapped.equals(wrappedItem.wrapped);
      }
      return this.wrapped.equals(other);
    }
  }
}
