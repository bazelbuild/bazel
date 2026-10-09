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
package com.google.devtools.build.lib.query2.common;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.common.options.OptionsParser;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CommonQueryOptions}. */
@RunWith(JUnit4.class)
public class CommonQueryOptionsTest {

  @Test
  public void defaultLineTerminatorIsNewline() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse();
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getLineTerminatorNull()).isFalse();
    assertThat(options.getLineTerminator()).isEqualTo("\n");
  }

  @Test
  public void nullOptionExpandsToLineTerminatorNull() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse("--null");
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getLineTerminatorNull()).isTrue();
    assertThat(options.getLineTerminator()).isEqualTo("\0");
  }

  @Test
  public void lineTerminatorNullOverridesNullOption() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(CommonQueryOptions.class).build();
    parser.parse("--null", "--line_terminator_null=false");
    CommonQueryOptions options = parser.getOptions(CommonQueryOptions.class);
    assertThat(options.getLineTerminatorNull()).isFalse();
    assertThat(options.getLineTerminator()).isEqualTo("\n");
  }
}
