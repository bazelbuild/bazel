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

import com.google.devtools.common.options.Converters.EmptyToNullStringConverter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class EmptyToNullStringConverterTest {

  private final EmptyToNullStringConverter converter = new EmptyToNullStringConverter();

  @Test
  public void emptyStringReturnsNull() throws OptionsParsingException {
    assertThat(converter.convert("")).isNull();
  }

  @Test
  public void literalNullStringPassesThrough() throws OptionsParsingException {
    assertThat(converter.convert("null")).isEqualTo("null");
  }

  @Test
  public void regularPathPassesThrough() throws OptionsParsingException {
    assertThat(converter.convert("/path/to/cert.pem")).isEqualTo("/path/to/cert.pem");
  }

  @Test
  public void arbitraryStringPassesThrough() throws OptionsParsingException {
    assertThat(converter.convert("some-value")).isEqualTo("some-value");
  }
}
