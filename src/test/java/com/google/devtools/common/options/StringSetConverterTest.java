// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.devtools.common.options.Converters.StringSetConverter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link StringSetConverter}. */
@RunWith(JUnit4.class)
public class StringSetConverterTest {

  @Test
  public void convertsValidValues() throws OptionsParsingException {
    StringSetConverter converter = new StringSetConverter("foo", "bar", "baz");
    assertThat(converter.convert("foo")).isEqualTo("foo");
    assertThat(converter.convert("bar")).isEqualTo("bar");
    assertThat(converter.convert("baz")).isEqualTo("baz");
  }

  @Test
  public void throwsExceptionOnInvalidValue() {
    StringSetConverter converter = new StringSetConverter("foo", "bar");
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> converter.convert("invalid"));
    assertThat(e).hasMessageThat().isEqualTo("Not one of [foo, bar]");
  }

  @Test
  public void handlesDuplicatesGracefully() throws OptionsParsingException {
    StringSetConverter converter = new StringSetConverter("foo", "bar", "foo");
    assertThat(converter.convert("foo")).isEqualTo("foo");
    assertThat(converter.convert("bar")).isEqualTo("bar");
    assertThat(converter.getTypeDescription()).isEqualTo("foo or bar");
  }

  @Test
  public void returnsCorrectTypeDescription() {
    StringSetConverter converter = new StringSetConverter("foo", "bar");
    assertThat(converter.getTypeDescription()).isEqualTo("foo or bar");
  }
}
