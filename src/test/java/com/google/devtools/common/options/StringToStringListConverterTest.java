// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link Converters.AssignmentToListOfValuesConverter}. */
@RunWith(JUnit4.class)
public class StringToStringListConverterTest {

  protected Converter<Map.Entry<String, List<String>>> converter =
      new Converters.StringToStringListConverter();

  @Test
  public void nameEqualsValue() throws Exception {
    assertThat(converter.convert("name=value"))
        .isEqualTo(Maps.immutableEntry("name", ImmutableList.of("value")));
  }

  @Test
  public void nameEqualsMultipleValues() throws Exception {
    assertThat(converter.convert("name=value1,value2"))
        .isEqualTo(Maps.immutableEntry("name", ImmutableList.of("value1", "value2")));
  }

  @Test
  public void nameEqualsNoValue_setsEmptyValue() throws Exception {
    assertThat(converter.convert("name="))
        .isEqualTo(Maps.immutableEntry("name", ImmutableList.of()));
  }

  @Test
  public void equalsValue_setsEmptyKey() throws Exception {
    assertThat(converter.convert("=value"))
        .isEqualTo(Maps.immutableEntry("", ImmutableList.of("value")));
  }

  @Test
  public void justValue_setsEmptyKey() throws Exception {
    assertThat(converter.convert("value"))
        .isEqualTo(Maps.immutableEntry("", ImmutableList.of("value")));
  }

  @Test
  public void noNameMultipleValues() throws Exception {
    assertThat(converter.convert("value1,value2"))
        .isEqualTo(Maps.immutableEntry("", ImmutableList.of("value1", "value2")));
  }
}
