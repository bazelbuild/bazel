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

package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link EnvVar.Converter}. */
@RunWith(JUnit4.class)
public class EnvVarConverterTest {

  private EnvVar.Converter converter = new EnvVar.Converter();

  private EnvVar convert(String input) throws Exception {
    return converter.convert(input);
  }

  @Test
  public void assignment() throws Exception {
    assertThat(convert("A=1")).isEqualTo(new EnvVar.Set("A", "1"));
    assertThat(convert("A=ABC")).isEqualTo(new EnvVar.Set("A", "ABC"));
    assertThat(convert("A=")).isEqualTo(new EnvVar.Set("A", ""));
    assertThat(convert("A=B,C=D")).isEqualTo(new EnvVar.Set("A", "B,C=D"));
  }

  @Test
  public void missingName() throws Exception {
    assertThat(convert("=NAME")).isEqualTo(new EnvVar.Unset("NAME"));
  }

  @Test
  public void missingValue() throws Exception {
    assertThat(convert("NAME")).isEqualTo(new EnvVar.Inherit("NAME"));
  }

  @Test
  public void reverseConversionForStarlark() throws Exception {
    assertThat(converter.reverseForStarlark(converter.convert("a"))).isEqualTo("a");
    assertThat(converter.reverseForStarlark(converter.convert("a=1"))).isEqualTo("a=1");
    assertThat(converter.reverseForStarlark(converter.convert("=a"))).isEqualTo("=a");
  }
}
