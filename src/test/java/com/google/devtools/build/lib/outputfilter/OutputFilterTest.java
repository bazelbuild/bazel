// Copyright 2019 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.outputfilter;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.events.OutputFilter;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the {@code --output_filter} option. */
@RunWith(JUnit4.class)
public class OutputFilterTest {

  @Test
  public void testOutputEverythingAlwaysTrue() {
    assertThat(OutputFilter.OUTPUT_EVERYTHING.showOutput("some tag")).isTrue();
    assertThat(OutputFilter.OUTPUT_EVERYTHING.showOutput("literally anything")).isTrue();
    assertThat(OutputFilter.OUTPUT_EVERYTHING.showOutput("even empty")).isTrue();
    assertThat(OutputFilter.OUTPUT_EVERYTHING.showOutput("")).isTrue();
  }

  @Test
  public void testOutputNothingAlwaysTrue() {
    assertThat(OutputFilter.OUTPUT_NOTHING.showOutput("some tag")).isFalse();
    assertThat(OutputFilter.OUTPUT_NOTHING.showOutput("literally anything")).isFalse();
    assertThat(OutputFilter.OUTPUT_NOTHING.showOutput("even empty")).isFalse();
    assertThat(OutputFilter.OUTPUT_NOTHING.showOutput("")).isFalse();
  }

  @Test
  public void testRegexpFilterShowOutputMatchTagReturnsTrue() {
    OutputFilter underTest =
        OutputFilter.RegexOutputFilter.forPattern(Pattern.compile("^//some/target"));
    assertThat(underTest.showOutput("//some/target")).isTrue();
  }

  @Test
  public void testRegexpFilterShowOutputNonMatchTagReturnsFalse() {
    OutputFilter underTest =
        OutputFilter.RegexOutputFilter.forPattern(Pattern.compile("^//some/target"));
    assertThat(underTest.showOutput("//not/some/target")).isFalse();
  }
}
