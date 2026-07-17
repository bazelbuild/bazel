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

package com.google.devtools.common.options;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.common.options.Converters.ByteSizeConverter;
import java.math.BigInteger;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ByteSizeConverterTest}. */
@RunWith(JUnit4.class)
public class ByteSizeConverterTest {

  ByteSizeConverter converter = new ByteSizeConverter();

  @Test
  public void empty() throws Exception {
    assertThrows(OptionsParsingException.class, () -> converter.convert(""));
  }

  @Test
  public void zero() throws Exception {
    assertThat(converter.convert("0")).isEqualTo(0L);
    assertThat(converter.convert("00")).isEqualTo(0L);
  }

  @Test
  public void negative() throws Exception {
    assertThrows(OptionsParsingException.class, () -> converter.convert("-1"));
  }

  @Test
  public void fractional() throws Exception {
    assertThrows(OptionsParsingException.class, () -> converter.convert("1.1"));
  }

  @Test
  public void nonDecimal() throws Exception {
    assertThrows(OptionsParsingException.class, () -> converter.convert("1f"));
  }

  @Test
  public void noSuffix() throws Exception {
    assertThat(converter.convert("123")).isEqualTo(123L);
  }

  @Test
  public void kiloSuffix() throws Exception {
    assertThat(converter.convert("123K")).isEqualTo(123 * 1024L);
  }

  @Test
  public void megaSuffix() throws Exception {
    assertThat(converter.convert("123M")).isEqualTo(123 * 1024L * 1024L);
  }

  @Test
  public void gigaSuffix() throws Exception {
    assertThat(converter.convert("123G")).isEqualTo(123 * 1024L * 1024L * 1024L);
  }

  @Test
  public void teraSuffix() throws Exception {
    assertThat(converter.convert("123T")).isEqualTo(123 * 1024L * 1024L * 1024L * 1024L);
  }

  @Test
  public void noSuffixOverflow() throws Exception {
    assertThrows(
        OptionsParsingException.class,
        () -> converter.convert(BigInteger.valueOf(2).pow(63).toString()));
  }

  @Test
  public void kiloOverflow() throws Exception {
    assertThrows(
        OptionsParsingException.class,
        () -> converter.convert(BigInteger.valueOf(2).pow(53) + "K"));
  }

  @Test
  public void megaOverflow() throws Exception {
    assertThrows(
        OptionsParsingException.class,
        () -> converter.convert(BigInteger.valueOf(2).pow(43) + "M"));
  }

  @Test
  public void gigaOverflow() throws Exception {
    assertThrows(
        OptionsParsingException.class,
        () -> converter.convert(BigInteger.valueOf(2).pow(33) + "G"));
  }

  @Test
  public void teraOverflow() throws Exception {
    assertThrows(
        OptionsParsingException.class,
        () -> converter.convert(BigInteger.valueOf(2).pow(23) + "T"));
  }
}
