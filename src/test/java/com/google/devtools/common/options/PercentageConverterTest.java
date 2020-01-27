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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static org.junit.Assert.assertEquals;

import com.google.devtools.common.options.Converters.PercentageConverter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link PercentageConverter}.
 */
@RunWith(JUnit4.class)
public final class PercentageConverterTest {

  private final PercentageConverter converter = new PercentageConverter();

  @Test
  public void shouldReturnIntegerValue() throws Exception {
    Integer percentage = 50;
    assertEquals(percentage, converter.convert(Integer.toString(percentage)));
  }

  @Test
  public void throwsExceptionWhenInputIsLessThanZero() {
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> converter.convert("-1"));
    assertThat(e).hasMessageThat().isEqualTo("'-1' should be >= 0");
  }

  @Test
  public void throwsExceptionWhenInputIsGreaterThanHundred() {
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> converter.convert("101"));
    assertThat(e).hasMessageThat().isEqualTo("'101' should be <= 100");
  }

  @Test
  public void throwsExceptionWhenInputIsNotANumber() {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> converter.convert("oops - not a number."));
    assertThat(e).hasMessageThat().isEqualTo("'oops - not a number.' is not an int");
  }
}
