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

import junit.framework.TestCase;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * A test for {@link Converters.CommaSeparatedOptionListConverter}.
 */
public class CommaSeparatedOptionListConverterTest
    extends TestCase {

  private Converter<List<String>> converter =
      new Converters.CommaSeparatedOptionListConverter();

  public void testEmptyStringYieldsEmptyList() throws Exception {
    assertEquals(Collections.emptyList(), converter.convert(""));
  }

  public void testCommaTwoEmptyStrings() throws Exception {
    assertEquals(Arrays.asList("", ""), converter.convert(","));
  }

  public void testLeadingCommaYieldsLeadingSpace() throws Exception {
    assertEquals(Arrays.asList("", "leading", "comma"),
                 converter.convert(",leading,comma"));
  }

  public void testTrailingCommaYieldsTrailingSpace() throws Exception {
    assertEquals(Arrays.asList("trailing", "comma", ""),
                 converter.convert("trailing,comma,"));
  }

  public void testSingleWord() throws Exception {
    assertEquals(Arrays.asList("lonely"), converter.convert("lonely"));
  }

  public void testMultiWords() throws Exception {
    assertEquals(Arrays.asList("one", "two", "three"),
                 converter.convert("one,two,three"));
  }

  public void testSpaceIsIgnored() throws Exception {
    assertEquals(Arrays.asList("one two three"),
                 converter.convert("one two three"));
  }

  public void testValueisUnmodifiable() throws Exception {
    try {
      converter.convert("value").add("other");
      fail("could modify value");
    } catch (UnsupportedOperationException expected) {}
  }

}
