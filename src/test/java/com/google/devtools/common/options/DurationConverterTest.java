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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.Converters.DurationConverter;
import java.time.Duration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DurationConverter}. */
@RunWith(JUnit4.class)
public class DurationConverterTest {

  @Test
  public void testDurationConverter_zero() throws OptionsParsingException {
    DurationConverter converter = new DurationConverter();

    assertThat(converter.convert("0")).isEqualTo(Duration.ZERO);
    assertThat(converter.convert("0d")).isEqualTo(Duration.ZERO);
    assertThat(converter.convert("0h")).isEqualTo(Duration.ZERO);
    assertThat(converter.convert("0m")).isEqualTo(Duration.ZERO);
    assertThat(converter.convert("0s")).isEqualTo(Duration.ZERO);
    assertThat(converter.convert("0ms")).isEqualTo(Duration.ZERO);
  }

  @Test
  public void testDurationConverter_basic() throws OptionsParsingException {
    DurationConverter converter = new DurationConverter();

    assertThat(converter.convert("10d")).isEqualTo(Duration.ofDays(10));
    assertThat(converter.convert("20h")).isEqualTo(Duration.ofHours(20));
    assertThat(converter.convert("30m")).isEqualTo(Duration.ofMinutes(30));
    assertThat(converter.convert("40s")).isEqualTo(Duration.ofSeconds(40));
    assertThat(converter.convert("50ms")).isEqualTo(Duration.ofMillis(50));
    assertThat(converter.convert("60ns")).isEqualTo(Duration.ofNanos(60));
  }

  @Test
  public void testDurationConverter_invalidInputs() {
    DurationConverter converter = new DurationConverter();

    assertThrows(OptionsParsingException.class, () -> converter.convert(""));

    assertThrows(OptionsParsingException.class, () -> converter.convert("-10d"));

    assertThrows(OptionsParsingException.class, () -> converter.convert("h"));

    assertThrows(OptionsParsingException.class, () -> converter.convert("1g"));
  }

  @Test
  public void testDurationConverter_maxValidValues() throws Exception {
    DurationConverter converter = new DurationConverter();

    assertThat(converter.convert("106751991167300d")).isEqualTo(Duration.ofDays(106751991167300L));
    assertThat(converter.convert("2562047788015215h"))
        .isEqualTo(Duration.ofHours(2562047788015215L));
    assertThat(converter.convert("153722867280912930m"))
        .isEqualTo(Duration.ofMinutes(153722867280912930L));
    assertThat(converter.convert(Long.MAX_VALUE + "s"))
        .isEqualTo(Duration.ofSeconds(Long.MAX_VALUE));
    assertThat(converter.convert(Long.MAX_VALUE + "ms"))
        .isEqualTo(Duration.ofMillis(Long.MAX_VALUE));
    assertThat(converter.convert(Long.MAX_VALUE + "ns"))
        .isEqualTo(Duration.ofNanos(Long.MAX_VALUE));
  }

  @Test
  public void testDurationConverter_arithmeticOverflow() {
    DurationConverter converter = new DurationConverter();

    for (String input :
        ImmutableList.of("106751991167301d", "2562047788015216h", "153722867280912931m")) {
      OptionsParsingException thrown =
          assertThrows(
              "input: " + input, OptionsParsingException.class, () -> converter.convert(input));
      assertThat(thrown).hasMessageThat().contains(input);
      assertThat(thrown).hasCauseThat().isInstanceOf(ArithmeticException.class);
    }
  }

  @Test
  public void testDurationConverter_numberFormatOverflow() {
    DurationConverter converter = new DurationConverter();

    for (String input :
        ImmutableList.of(
            "9223372036854775808d",
            "9223372036854775808h",
            "9223372036854775808m",
            "9223372036854775808s",
            "9223372036854775808ms",
            "9223372036854775808ns")) {
      OptionsParsingException thrown =
          assertThrows(
              "input: " + input, OptionsParsingException.class, () -> converter.convert(input));
      assertThat(thrown).hasMessageThat().contains(input);
      assertThat(thrown).hasCauseThat().isInstanceOf(NumberFormatException.class);
    }
  }
}
