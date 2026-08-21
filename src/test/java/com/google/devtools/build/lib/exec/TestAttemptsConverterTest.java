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

import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TestAttemptsConverterTest {
  private final ExecutionOptions.TestAttemptsConverter converter =
      new ExecutionOptions.TestAttemptsConverter();

  @Test
  public void globalStableFlakyPair() throws Exception {
    PerLabelOptions options = converter.convert("1,3");
    assertThat(options.getOptions()).containsExactly("1,3");
    assertThat(FlakyTestAttempts.parse(options.getOptions().get(0)).getAttempts(false)).isEqualTo(1);
    assertThat(FlakyTestAttempts.parse(options.getOptions().get(0)).getAttempts(true)).isEqualTo(3);
  }

  @Test
  public void globalSingleInteger() throws Exception {
    PerLabelOptions options = converter.convert("3");
    assertThat(options.getOptions()).containsExactly("3");
  }

  @Test
  public void regexOverrideAllowsSingleInteger() throws Exception {
    PerLabelOptions options = converter.convert("//foo/.*@3");
    assertThat(options.getOptions()).containsExactly("3");
  }

  @Test
  public void regexOverrideRejectsStableFlakyPair() {
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> converter.convert("//foo/.*@1,3"));
    assertThat(e).hasMessageThat().contains("per-target overrides must use a single integer");
  }
}
