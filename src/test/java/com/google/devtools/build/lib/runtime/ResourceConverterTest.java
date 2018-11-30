// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.function.Supplier;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link ResourceConverter}. */
@RunWith(JUnit4.class)
public class ResourceConverterTest {

  static class FakeResourceConverter extends ResourceConverter {

    FakeResourceConverter(Supplier<Integer> supplier1, Supplier<Integer> supplier2)
        throws OptionsParsingException {
      super(
          ImmutableMap.<String, Supplier<Integer>>builder()
              .put("keyword1", supplier1)
              .put("keyword2", supplier2)
              .build());
    }
  }

  private FakeResourceConverter fakeResourceConverter;

  @Before
  public final void setUp() throws OptionsParsingException {
    // Input of "keyword1" returns 5; "keyword2" returns 10.
    fakeResourceConverter = new FakeResourceConverter(() -> 5, () -> 10);
  }

  @Test
  public void testResourceConverterParsesInt() throws OptionsParsingException {
    assertThat(fakeResourceConverter.convert("6")).isEqualTo(6);
  }

  @Test
  public void testResourceConverterFailsOnFloat() {
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> fakeResourceConverter.convert(".5"));
    assertThat(thrown).hasMessageThat().contains("not an int");
  }

  @Test
  public void testResourceConverterLinksKeywordToFunction() throws Exception {
    assertThat(fakeResourceConverter.convert("keyword1")).isEqualTo(5);
  }

  @Test
  public void testResourceConverterLinksKeywordToFunction2() throws Exception {
    assertThat(fakeResourceConverter.convert("keyword2")).isEqualTo(10);
  }

  @Test
  public void testResourceConverterMinusValueWorks() throws Exception {
    assertThat(fakeResourceConverter.convert("keyword1-1")).isEqualTo(4);
  }

  @Test
  public void testResourceConverterTimesValueWorks() throws Exception {
    assertThat(fakeResourceConverter.convert("keyword2*.5")).isEqualTo(5);
  }

  @Test
  public void testResourceConverterFailsOnWrongKeyword() {
    OptionsParsingException thrown =
        assertThrows(
            OptionsParsingException.class, () -> fakeResourceConverter.convert("invalid_keyword"));
    assertThat(thrown)
        .hasMessageThat()
        .isEqualTo(
            "Parameter 'invalid_keyword' does not follow correct syntax. "
                + "This flag takes [keyword1|keyword2][-|*]<float>.");
  }

  @Test
  public void testResourceConverterCalculatedValueRounds() throws Exception {
    assertThat(fakeResourceConverter.convert("keyword1*.51")).isEqualTo(3);
  }

  @Test
  public void testResourceConverterFailsOnAlmostValidKeyword() {
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, () -> fakeResourceConverter.convert("keyword"));
    assertThat(thrown).hasMessageThat().contains("does not follow correct syntax");
  }

  @Test
  public void testResourceConverterFailsOnInvalidOperator() {
    OptionsParsingException thrown =
        assertThrows(
            OptionsParsingException.class, () -> fakeResourceConverter.convert("keyword1/2"));
    assertThat(thrown).hasMessageThat().contains("does not contain a valid operator");
  }

  @Test
  public void testResourceConverterFailsWithInvalidKeywords() {
    class InvalidResourceConverter extends ResourceConverter {
      // Create a resource converter with invalid keyword pair, "keyword" and "keyword1"
      private InvalidResourceConverter() throws OptionsParsingException {
        super(
            ImmutableMap.<String, Supplier<Integer>>builder()
                .put("keyword", () -> 1)
                .put("keyword1", () -> 2)
                .build());
      }
    }
    OptionsParsingException thrown =
        assertThrows(OptionsParsingException.class, InvalidResourceConverter::new);
    assertThat(thrown)
        .hasMessageThat()
        .isEqualTo("Keywords (keyword,keyword1) must not be starting substrings of each other.");
  }
}
