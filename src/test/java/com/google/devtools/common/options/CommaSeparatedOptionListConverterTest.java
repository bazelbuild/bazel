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

import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link Converters.CommaSeparatedOptionListConverter}. */
@RunWith(JUnit4.class)
public class CommaSeparatedOptionListConverterTest {

  private Converter<List<String>> converter =
      new Converters.CommaSeparatedOptionListConverter();

  @Test
  public void emptyStringYieldsEmptyList() throws Exception {
    assertThat(converter.convert("")).isEmpty();
  }

  @Test
  public void commaTwoEmptyStrings() throws Exception {
    assertThat(converter.convert(",")).containsExactly("", "").inOrder();
  }

  @Test
  public void leadingCommaYieldsLeadingSpace() throws Exception {
    assertThat(converter.convert(",leading,comma"))
        .containsExactly("", "leading", "comma").inOrder();
  }

  @Test
  public void trailingCommaYieldsTrailingSpace() throws Exception {
    assertThat(converter.convert("trailing,comma,"))
        .containsExactly("trailing", "comma", "").inOrder();
  }

  @Test
  public void singleWord() throws Exception {
    assertThat(converter.convert("lonely")).containsExactly("lonely");
  }

  @Test
  public void multiWords() throws Exception {
    assertThat(converter.convert("one,two,three"))
        .containsExactly("one", "two", "three").inOrder();
  }

  @Test
  public void spaceIsIgnored() throws Exception {
    assertThat(converter.convert("one two three")).containsExactly("one two three");
  }

  @Test
  public void valueisUnmodifiable() throws Exception {
    assertThrows(
        "could modify value",
        UnsupportedOperationException.class,
        () -> converter.convert("value").add("other"));
  }

}
