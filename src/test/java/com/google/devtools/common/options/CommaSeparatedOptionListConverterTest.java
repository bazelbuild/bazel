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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A test for {@link Converters.CommaSeparatedOptionListConverter}.
 */
@RunWith(JUnit4.class)
public class CommaSeparatedOptionListConverterTest {

  private Converter<List<String>> converter =
      new Converters.CommaSeparatedOptionListConverter();

  @Test
  public void emptyStringYieldsEmptyList() throws Exception {
    assertEquals(Collections.emptyList(), converter.convert(""));
  }

  @Test
  public void commaTwoEmptyStrings() throws Exception {
    assertEquals(Arrays.asList("", ""), converter.convert(","));
  }

  @Test
  public void leadingCommaYieldsLeadingSpace() throws Exception {
    assertEquals(Arrays.asList("", "leading", "comma"),
                 converter.convert(",leading,comma"));
  }

  @Test
  public void trailingCommaYieldsTrailingSpace() throws Exception {
    assertEquals(Arrays.asList("trailing", "comma", ""),
                 converter.convert("trailing,comma,"));
  }

  @Test
  public void singleWord() throws Exception {
    assertEquals(Arrays.asList("lonely"), converter.convert("lonely"));
  }

  @Test
  public void multiWords() throws Exception {
    assertEquals(Arrays.asList("one", "two", "three"),
                 converter.convert("one,two,three"));
  }

  @Test
  public void spaceIsIgnored() throws Exception {
    assertEquals(Arrays.asList("one two three"),
                 converter.convert("one two three"));
  }

  @Test
  public void valueisUnmodifiable() throws Exception {
    try {
      converter.convert("value").add("other");
      fail("could modify value");
    } catch (UnsupportedOperationException expected) {}
  }

}
