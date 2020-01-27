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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests to exercise the functionality of {@link ConverterTester}. */
@RunWith(JUnit4.class)
public final class ConverterTesterTest {

  @Test
  public void construction_throwsAssertionErrorIfConverterCreationFails() throws Exception {
    try {
      new ConverterTester(UnconstructableConverter.class);
    } catch (AssertionError expected) {
      assertThat(expected) // AssertionError
          .hasCauseThat() // InvocationTargetException
          .hasCauseThat() // UnsupportedOperationException
          .hasMessageThat()
          .contains("YOU CAN'T MAKE ME!");
      return;
    }
    fail("expected tester creation to fail");
  }

  /** Test converter for construction_throwsAssertionErrorIfConverterCreationFails. */
  public static final class UnconstructableConverter implements Converter<String> {
    public UnconstructableConverter() {
      throw new UnsupportedOperationException("YOU CAN'T MAKE ME!");
    }

    @Override
    public String convert(String input) throws OptionsParsingException {
      return input;
    }

    @Override
    public String getTypeDescription() {
      return "anything, if you can get an instance";
    }
  }

  @Test
  public void getConverterClass_returnsConstructorArg() throws Exception {
    ConverterTester tester = new ConverterTester(Converters.BooleanConverter.class);
    assertThat(tester.getConverterClass()).isEqualTo(Converters.BooleanConverter.class);
  }

  @Test
  public void hasTestForInput_returnsTrueIffInputPassedToAddEqualityGroup() throws Exception {
    ConverterTester tester =
        new ConverterTester(Converters.DoubleConverter.class)
            .addEqualityGroup("1.0", "1", "1.00")
            .addEqualityGroup("2");

    assertThat(tester.hasTestForInput("1.0")).isTrue();
    assertThat(tester.hasTestForInput("1")).isTrue();
    assertThat(tester.hasTestForInput("1.00")).isTrue();
    assertThat(tester.hasTestForInput("2")).isTrue();

    assertThat(tester.hasTestForInput("3")).isFalse();
    assertThat(tester.hasTestForInput("1.000")).isFalse();
    assertThat(tester.hasTestForInput("not a double")).isFalse();
  }

  @Test
  public void addEqualityGroup_throwsIfConversionFails() throws Exception {
    ConverterTester tester =
        new ConverterTester(ThrowingConverter.class)
            .addEqualityGroup("okay")
            .addEqualityGroup("also okay", "pretty fine");
    try {
      tester.addEqualityGroup("wrong");
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("\"wrong\"");
      assertThat(expected).hasCauseThat().hasMessageThat().contains("HOW DARE YOU");
      return;
    }
    fail("expected addEqualityGroup to fail");
  }

  /** Test converter for addEqualityGroup_throwsIfConversionFails. */
  public static final class ThrowingConverter implements Converter<String> {
    @Override
    public String convert(String input) throws OptionsParsingException {
      if ("wrong".equals(input)) {
        throw new OptionsParsingException("HOW DARE YOU");
      }
      return input;
    }

    @Override
    public String getTypeDescription() {
      return "just don't give the wrong answer";
    }
  }

  @Test
  public void testConvert_passesWhenAllInstancesObeyEqualsAndItemsOnlyEqualToOthersInSameGroup() {
        new ConverterTester(Converters.DoubleConverter.class)
            .addEqualityGroup("1.0", "1", "1.00")
            .addEqualityGroup("2", "2", "2.0000", "2.0", "+2")
            .addEqualityGroup("3")
            .addEqualityGroup("3.1415")
            .testConvert();
  }

  @Test
  public void testConvert_testsHashCodeConsistencyForConvertedInstance() {
    ConverterTester tester =
        new ConverterTester(InconsistentHashCodeConverter.class)
            .addEqualityGroup("input doesn't matter");
    try {
      tester.testConvert();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("\"input doesn't matter\"");
      assertThat(expected).hasMessageThat().contains("hashCode");
      assertThat(expected).hasMessageThat().contains("must be consistent");
      return;
    }
    fail("expected the tester to notice the bad hash code implementation");
  }

  /** A class with a badly implemented hashCode which is not consistent across calls. */
  public static final class InconsistentHashCode {
    @Override
    public boolean equals(Object other) {
      return other instanceof InconsistentHashCode;
    }

    private int howManyTimesHaveIBeenHashedAlready = 0;

    @Override
    public int hashCode() {
      howManyTimesHaveIBeenHashedAlready += 1;
      return howManyTimesHaveIBeenHashedAlready;
    }
  }

  /** Test converter for testConvert_testsHashCodeConsistencyForConvertedInstance. */
  public static final class InconsistentHashCodeConverter
      implements Converter<InconsistentHashCode> {
    @Override
    public InconsistentHashCode convert(String input) throws OptionsParsingException {
      return new InconsistentHashCode();
    }

    @Override
    public String getTypeDescription() {
      return "anything, I don't even look at it";
    }
  }

  @Test
  public void testConvert_testsHashCodeConsistencyForSameConverter() {
    ConverterTester tester =
        new ConverterTester(IncrementingHashCodeConverter.class)
            .addEqualityGroup("meaningless input");
    try {
      tester.testConvert();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("\"meaningless input\"");
      assertThat(expected).hasMessageThat().contains("consistent hashCode");
      assertThat(expected).hasMessageThat().contains("same Converter");
      return;
    }
    fail("expected the tester to notice the mismatched hash codes");
  }

  /** A class with a configurable hashCode set in the constructor. */
  public static final class SettableHashCode {
    private final int hashCode;

    public SettableHashCode(int hashCode) {
      this.hashCode = hashCode;
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof SettableHashCode;
    }

    @Override
    public int hashCode() {
      return hashCode;
    }
  }

  /** Test converter for testConvert_testsHashCodeConsistencyForSameConverter. */
  public static final class IncrementingHashCodeConverter
      implements Converter<SettableHashCode> {
    private int howManyInstancesHaveIMadeAlready = 0;

    @Override
    public SettableHashCode convert(String input) throws OptionsParsingException {
      howManyInstancesHaveIMadeAlready += 1;
      return new SettableHashCode(howManyInstancesHaveIMadeAlready);
    }

    @Override
    public String getTypeDescription() {
      return "whatever, I'm pretty much just going to ignore it";
    }
  }

  @Test
  public void testConvert_testsHashCodeConsistencyForDifferentConverters() {
    ConverterTester tester =
        new ConverterTester(StaticIncrementingHashCodeConverter.class)
            .addEqualityGroup("some kind of input");
    try {
      tester.testConvert();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("\"some kind of input\"");
      assertThat(expected).hasMessageThat().contains("consistent hashCode");
      assertThat(expected).hasMessageThat().contains("different Converter");
      return;
    }
    fail("expected the tester to notice the mismatched hash codes");
  }

  /** Test converter for testConvert_testsHashCodeConsistencyForDifferentConverters. */
  public static final class StaticIncrementingHashCodeConverter
      implements Converter<SettableHashCode> {
    private static int howManyInstancesHaveIMadeAlready = 0;

    private final int hashCode;

    public StaticIncrementingHashCodeConverter() {
      howManyInstancesHaveIMadeAlready += 1;
      this.hashCode = howManyInstancesHaveIMadeAlready;
    }

    @Override
    public SettableHashCode convert(String input) throws OptionsParsingException {
      return new SettableHashCode(hashCode);
    }

    @Override
    public String getTypeDescription() {
      return "a string or null, I'm easy";
    }
  }


  @Test
  public void testConvert_testsSelfEqualityForConvertedInstance() {
    ConverterTester tester =
        new ConverterTester(SelfLoathingConverter.class)
            .addEqualityGroup("self-loathing");
    try {
      tester.testConvert();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("\"self-loathing\"");
      assertThat(expected).hasMessageThat().contains("must be Object#equals to itself");
      return;
    }
    fail("expected the tester to notice the bad equals implementation");
  }

  /** A class which is equal to every instance of its class except itself. */
  public static final class SelfLoathingObject {
    @Override
    public boolean equals(Object other) {
      return other instanceof SelfLoathingObject && other != this;
    }

    @Override
    public int hashCode() {
      return 4; // chosen by fair hashing algorithm
    }
  }

  /** Test converter for testConvert_testsSelfEqualityForConvertedInstance. */
  public static final class SelfLoathingConverter implements Converter<SelfLoathingObject> {
    @Override
    public SelfLoathingObject convert(String input) throws OptionsParsingException {
      return new SelfLoathingObject();
    }

    @Override
    public String getTypeDescription() {
      return "whatever... why even ask me to convert anything..............";
    }
  }

  @Test
  public void testConvert_testsEqualityForSameConverter() {
    ConverterTester tester =
        new ConverterTester(CountingConverter.class)
            .addEqualityGroup("countables");
    try {
      tester.testConvert();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("\"countables\"");
      assertThat(expected).hasMessageThat().contains("equal to itself");
      assertThat(expected).hasMessageThat().contains("same Converter");
      return;
    }
    fail("expected the tester to notice the converter giving unequal objects");
  }

  /** Test converter for testConvert_testsEqualityForSameConverter. */
  public static final class CountingConverter implements Converter<Integer> {
    private int howManyInstancesHaveIMadeAlready = 0;

    @Override
    public Integer convert(String input) throws OptionsParsingException {
      howManyInstancesHaveIMadeAlready += 1;
      return howManyInstancesHaveIMadeAlready;
    }

    @Override
    public String getTypeDescription() {
      return "I can count anything!";
    }
  }

  @Test
  public void testConvert_testsEqualityForDifferentConverters() {
    ConverterTester tester =
        new ConverterTester(StaticCountingConverter.class)
            .addEqualityGroup("words I like");
    try {
      tester.testConvert();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("\"words I like\"");
      assertThat(expected).hasMessageThat().contains("equal to itself");
      assertThat(expected).hasMessageThat().contains("different Converter");
      return;
    }
    fail("expected the tester to notice the converters giving unequal objects");
  }

  /** Test converter for testConvert_testsEqualityForDifferentConverters. */
  public static final class StaticCountingConverter implements Converter<Integer> {
    private static int howManyInstancesHaveIMadeAlready = 0;

    private final int output;

    public StaticCountingConverter() {
      howManyInstancesHaveIMadeAlready += 1;
      this.output = howManyInstancesHaveIMadeAlready;
    }

    @Override
    public Integer convert(String input) throws OptionsParsingException {
      return output;
    }

    @Override
    public String getTypeDescription() {
      return "your favorite text";
    }
  }

  @Test
  public void testConvert_testsEqualityForItemsInSameGroup() {
    ConverterTester tester =
        new ConverterTester(Converters.DoubleConverter.class)
            .addEqualityGroup("+1.000", "2.30");
    try {
      tester.testConvert();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("\"+1.000\"");
      assertThat(expected).hasMessageThat().contains("\"2.30\"");
      return;
    }
    fail("expected the tester to notice the two non-equal conversion results in the same group");
  }

  @Test
  public void testConvert_testsNonEqualityForItemsInDifferentGroups() {
    ConverterTester tester =
        new ConverterTester(Converters.DoubleConverter.class)
            .addEqualityGroup("+1.000")
            .addEqualityGroup("1.0");
    try {
      tester.testConvert();
    } catch (AssertionError expected) {
      assertThat(expected).hasMessageThat().contains("\"+1.000\"");
      assertThat(expected).hasMessageThat().contains("\"1.0\"");
      return;
    }
    fail("expected the tester to notice the two equal conversion results in different groups");
  }
}
