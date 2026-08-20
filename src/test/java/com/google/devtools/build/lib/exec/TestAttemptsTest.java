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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.PerLabelOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.exec.ExecutionOptions.TestAttemptsConverter;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link com.google.devtools.build.lib.exec.ExecutionOptions.TestAttemptsConverter}. */
@RunWith(JUnit4.class)
public class TestAttemptsTest {

  private TestAttemptsConverter converter;

  @Before
  public void setUp() {
    converter = new TestAttemptsConverter();
  }

  @Test
  public void testFlakyTestAttemptsWithAtSignInLabel() throws Exception {
    PerLabelOptions options = converter.convert("//foo:v@lid-target@3");
    assertThat(options.isIncluded(Label.parseCanonicalUnchecked("//foo:v@lid-target"))).isTrue();
    assertThat(options.getOptions()).isEqualTo(ImmutableList.of("3"));
  }

  @Test
  public void testFlakyTestAttemptsWithExternalRepoLabel() throws Exception {
    PerLabelOptions options = converter.convert("@repo//foo:bar@3");
    assertThat(options.isIncluded(Label.parseCanonicalUnchecked("@repo//foo:bar"))).isTrue();
    assertThat(options.getOptions()).isEqualTo(ImmutableList.of("3"));
  }

  @Test
  public void testFlakyTestAttemptsDefaultOrInteger() throws Exception {
    PerLabelOptions defaultOptions = converter.convert("default");
    assertThat(defaultOptions.getOptions()).isEqualTo(ImmutableList.of("default"));

    PerLabelOptions intOptions = converter.convert("3");
    assertThat(intOptions.getOptions()).isEqualTo(ImmutableList.of("3"));
  }

  @Test
  public void testFlakyTestAttemptsInvalidIntegerThrows() {
    assertThrows(
        OptionsParsingException.class, () -> converter.convert("//foo:v@lid-target@invalid"));
  }
}
