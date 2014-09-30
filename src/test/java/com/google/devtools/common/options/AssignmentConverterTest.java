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

import com.google.common.collect.Maps;

import junit.framework.TestCase;

import java.util.Map;

/**
 * Test for {@link Converters.AssignmentConverter} and
 * {@link Converters.OptionalAssignmentConverter}.
 */
public abstract class AssignmentConverterTest extends TestCase {

  protected Converter<Map.Entry<String, String>> converter = null;

  protected abstract void setConverter();

  protected Map.Entry<String, String> convert(String input) throws Exception {
    return converter.convert(input);
  }

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    setConverter();
  }

  public void testAssignment() throws Exception {
    assertEquals(Maps.immutableEntry("A", "1"), convert("A=1"));
    assertEquals(Maps.immutableEntry("A", "ABC"), convert("A=ABC"));
    assertEquals(Maps.immutableEntry("A", ""), convert("A="));
  }

  public void testMissingName() throws Exception {
    try {
      convert("=VALUE");
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }

  public void testEmptyString() throws Exception {
    try {
      convert("");
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }

  abstract public void testMissingValue() throws Exception;

  public static class MandatoryAssignmentConverterTest extends AssignmentConverterTest {

    @Override
    protected void setConverter() {
      converter = new Converters.AssignmentConverter();
    }

    @Override
    public void testMissingValue() throws Exception {
      try {
        convert("NAME");
        fail();
      } catch (OptionsParsingException e) {
        // expected.
      }
    }
  }

  public static class OptionalAssignmentConverterTest extends AssignmentConverterTest {

    @Override
    protected void setConverter() {
      converter = new Converters.OptionalAssignmentConverter();
    }

    @Override
    public void testMissingValue() throws Exception {
      assertEquals(Maps.immutableEntry("NAME", null), convert("NAME"));
    }
  }
}
