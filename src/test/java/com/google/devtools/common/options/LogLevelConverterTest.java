// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.common.options.Converters.LogLevelConverter;

import junit.framework.TestCase;

import java.util.logging.Level;

/**
 * A test for {@link LogLevelConverter}.
 */
public class LogLevelConverterTest extends TestCase {

  private LogLevelConverter converter = new LogLevelConverter();

  public void testConvertsIntsToLevels() throws OptionsParsingException {
    int levelId = 0;
    for (Level level : LogLevelConverter.LEVELS) {
      assertEquals(level, converter.convert(Integer.toString(levelId++)));
    }
  }

  public void testThrowsExceptionWhenInputIsNotANumber() {
    try {
      converter.convert("oops - not a number.");
      fail();
    } catch (OptionsParsingException e) {
      assertEquals("Not a log level: oops - not a number.", e.getMessage());
    }
  }

  public void testThrowsExceptionWhenInputIsInvalidInteger() {
    for (int example : new int[] {-1, 100, 50000}) {
      try {
        converter.convert(Integer.toString(example));
        fail();
      } catch (OptionsParsingException e) {
        String expected = "Not a log level: " + Integer.toString(example);
        assertEquals(expected, e.getMessage());
      }
    }
  }

}
