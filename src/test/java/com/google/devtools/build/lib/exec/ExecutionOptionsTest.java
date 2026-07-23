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

import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ExecutionOptionsTest {
  private final ExecutionOptions.TestAttemptsConverter converter =
      new ExecutionOptions.TestAttemptsConverter();

  @Test
  public void flakyTestAttempts_withAtInLabel() throws Exception {
    PerLabelOptions options = converter.convert("//foo:v@lid-target@3");

    assertThat(options.isIncluded(Label.parseCanonicalUnchecked("//foo:v@lid-target"))).isTrue();
    assertThat(options.getOptions()).containsExactly("3");
  }

  @Test
  public void flakyTestAttempts_withAtInLabelAndDefaultAttempts() throws Exception {
    PerLabelOptions options = converter.convert("//foo:v@lid-target@default");

    assertThat(options.isIncluded(Label.parseCanonicalUnchecked("//foo:v@lid-target"))).isTrue();
    assertThat(options.getOptions()).containsExactly("default");
  }
}
