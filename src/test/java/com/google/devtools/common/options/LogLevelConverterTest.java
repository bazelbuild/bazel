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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.common.options.Converters.LogLevelConverter;
import java.util.logging.Level;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link LogLevelConverter}. */
@RunWith(JUnit4.class)
public class LogLevelConverterTest {

  private LogLevelConverter converter = new LogLevelConverter();

  @Test
  public void convertsIntsToLevels() throws OptionsParsingException {
    int levelId = 0;
    for (Level level : LogLevelConverter.LEVELS) {
      assertThat(converter.convert(Integer.toString(levelId++))).isEqualTo(level);
    }
  }

  @Test
  public void throwsExceptionWhenInputIsNotANumber() {
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> converter.convert("oops - not a number."));
    assertThat(e).hasMessageThat().isEqualTo("Not a log level: oops - not a number.");
  }

  @Test
  public void throwsExceptionWhenInputIsInvalidInteger() {
    for (int example : new int[] {-1, 100, 50000}) {
      OptionsParsingException e =
          assertThrows(
              OptionsParsingException.class, () -> converter.convert(Integer.toString(example)));
      String expected = "Not a log level: " + Integer.toString(example);
      assertThat(e).hasMessageThat().isEqualTo(expected);
    }
  }

}
