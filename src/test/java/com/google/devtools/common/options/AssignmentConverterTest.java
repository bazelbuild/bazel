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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.common.collect.Maps;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Map;

/**
 * Test for {@link Converters.AssignmentConverter} and
 * {@link Converters.OptionalAssignmentConverter}.
 */
public abstract class AssignmentConverterTest {

  protected Converter<Map.Entry<String, String>> converter = null;

  protected abstract void setConverter();

  protected Map.Entry<String, String> convert(String input) throws Exception {
    return converter.convert(input);
  }

  @Before
  public void setUp() throws Exception {
    setConverter();
  }

  @Test
  public void assignment() throws Exception {
    assertEquals(Maps.immutableEntry("A", "1"), convert("A=1"));
    assertEquals(Maps.immutableEntry("A", "ABC"), convert("A=ABC"));
    assertEquals(Maps.immutableEntry("A", ""), convert("A="));
  }

  @Test
  public void missingName() throws Exception {
    try {
      convert("=VALUE");
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }

  @Test
  public void emptyString() throws Exception {
    try {
      convert("");
      fail();
    } catch (OptionsParsingException e) {
      // expected.
    }
  }


  @RunWith(JUnit4.class)
  public static class MandatoryAssignmentConverterTest extends AssignmentConverterTest {

    @Override
    protected void setConverter() {
      converter = new Converters.AssignmentConverter();
    }

    @Test
    public void missingValue() throws Exception {
      try {
        convert("NAME");
        fail();
      } catch (OptionsParsingException e) {
        // expected.
      }
    }
  }

  @RunWith(JUnit4.class)
  public static class OptionalAssignmentConverterTest extends AssignmentConverterTest {

    @Override
    protected void setConverter() {
      converter = new Converters.OptionalAssignmentConverter();
    }

    @Test
    public void missingValue() throws Exception {
      assertEquals(Maps.immutableEntry("NAME", null), convert("NAME"));
    }
  }
}
