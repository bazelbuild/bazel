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

package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class FlakyTestAttemptsTest {
  @Test
  public void parseDefault() throws Exception {
    assertThat(FlakyTestAttempts.parse("default")).isEqualTo(FlakyTestAttempts.DEFAULT);
  }

  @Test
  public void parseSingleIntegerAppliesToBothBuckets() throws Exception {
    FlakyTestAttempts attempts = FlakyTestAttempts.parse("3");
    assertThat(attempts.getStableAttempts()).isEqualTo(3);
    assertThat(attempts.getFlakyAttempts()).isEqualTo(3);
    assertThat(attempts.getAttempts(false)).isEqualTo(3);
    assertThat(attempts.getAttempts(true)).isEqualTo(3);
  }

  @Test
  public void parseStableFlakyPair() throws Exception {
    FlakyTestAttempts attempts = FlakyTestAttempts.parse("1,3");
    assertThat(attempts.getStableAttempts()).isEqualTo(1);
    assertThat(attempts.getFlakyAttempts()).isEqualTo(3);
    assertThat(attempts.getAttempts(false)).isEqualTo(1);
    assertThat(attempts.getAttempts(true)).isEqualTo(3);
  }

  @Test
  public void toCanonicalStringUsesSingleIntegerWhenEqual() {
    assertThat(new FlakyTestAttempts(3, 3).toCanonicalString()).isEqualTo("3");
    assertThat(new FlakyTestAttempts(1, 3).toCanonicalString()).isEqualTo("1,3");
  }

  @Test
  public void parseRejectsTooManyValues() {
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> FlakyTestAttempts.parse("1,2,3"));
    assertThat(e).hasMessageThat().contains("two comma-separated integers");
  }

  @Test
  public void parseRejectsEmptyValues() {
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> FlakyTestAttempts.parse("1,"));
    assertThat(e).hasMessageThat().contains("two comma-separated integers");
  }

  @Test
  public void parseRejectsOutOfRangeValues() {
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> FlakyTestAttempts.parse("0,3"));
    assertThat(e).hasMessageThat().contains("should be >=");
  }
}
