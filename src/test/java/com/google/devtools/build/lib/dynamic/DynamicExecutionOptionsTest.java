// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.dynamic;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.dynamic.DynamicExecutionOptions.SignalListConverter;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class DynamicExecutionOptionsTest {
  @Test
  public void testSignalNameConverter_convertsIntegers() throws OptionsParsingException {
    SignalListConverter converter = new SignalListConverter();
    assertThat(converter.convert("9", null)).containsExactly(9);
    assertThat(converter.convert("1,12,64", null)).containsExactly(1, 12, 64);
    assertThat(converter.convert("1,2,1,4", null)).containsExactly(1, 2, 4);
    assertThat(converter.convert("9", null)).containsExactly(9);
    assertThat(converter.convert("  12 , 14\t", null)).containsExactly(12, 14);
  }

  @Test
  public void testSignalNameConverter_badInputs() throws OptionsParsingException {
    SignalListConverter converter = new SignalListConverter();
    assertThat(converter.convert(null, null)).isEmpty();
    assertThat(converter.convert("null", null)).isEmpty();
    assertThrows(OptionsParsingException.class, () -> converter.convert("\t  ", null));
    assertThrows(OptionsParsingException.class, () -> converter.convert("", null));
    assertThrows(OptionsParsingException.class, () -> converter.convert("-1", null));
    assertThrows(OptionsParsingException.class, () -> converter.convert("5,,6", null));
    assertThrows(OptionsParsingException.class, () -> converter.convert("5.3", null));
    assertThrows(OptionsParsingException.class, () -> converter.convert("", null));
    assertThrows(OptionsParsingException.class, () -> converter.convert("SIGTERM", null));
  }
}
