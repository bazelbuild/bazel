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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.Maps;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link Converters.AssignmentConverter} and {@link
 * Converters.OptionalAssignmentConverter}.
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
    assertThat(convert("A=1")).isEqualTo(Maps.immutableEntry("A", "1"));
    assertThat(convert("A=ABC")).isEqualTo(Maps.immutableEntry("A", "ABC"));
    assertThat(convert("A=")).isEqualTo(Maps.immutableEntry("A", ""));
    assertThat(convert("A=B,C=D")).isEqualTo(Maps.immutableEntry("A", "B,C=D"));
  }

  @Test
  public void missingName() throws Exception {
    assertThrows(OptionsParsingException.class, () -> convert("=VALUE"));
  }

  @Test
  public void emptyString() throws Exception {
    assertThrows(OptionsParsingException.class, () -> convert(""));
  }


  @RunWith(JUnit4.class)
  public static class MandatoryAssignmentConverterTest extends AssignmentConverterTest {

    @Override
    protected void setConverter() {
      converter = new Converters.AssignmentConverter();
    }

    @Test
    public void missingValue() throws Exception {
      assertThrows(OptionsParsingException.class, () -> convert("NAME"));
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
      assertThat(convert("NAME")).isEqualTo(Maps.immutableEntry("NAME", null));
    }
  }
}
