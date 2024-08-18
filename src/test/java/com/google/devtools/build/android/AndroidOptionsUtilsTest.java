// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;

import com.beust.jcommander.JCommander;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class AndroidOptionsUtilsTest {
  private TestOptions options;
  private JCommander jc;

  @Before
  public void setUp() {
    options = new TestOptions();
    jc = new JCommander(options);
  }

  @Test
  public void testNormalizeFalseBoolean() {
    String[] args = {"--nofoo"};
    String[] normalizedArgs = AndroidOptionsUtils.normalizeBooleanOptions(options, args);
    assertThat(normalizedArgs[0]).isEqualTo("--foo=false");
    jc.parse(normalizedArgs);
    assertThat(options.foo).isFalse();
  }

  @Test
  public void testNormalizeTrueBoolean() {
    String[] args = {"--foo"};
    String[] normalizedArgs = AndroidOptionsUtils.normalizeBooleanOptions(options, args);
    assertThat(normalizedArgs[0]).isEqualTo("--foo=true");
    jc.parse(normalizedArgs);
    assertThat(options.foo).isTrue();
  }

  @Test
  public void testNormalizeNoBoolean() {
    String[] args = {"--normalize"};
    String[] normalizedArgs = AndroidOptionsUtils.normalizeBooleanOptions(options, args);
    assertThat(normalizedArgs[0]).isEqualTo("--normalize=true");
    jc.parse(normalizedArgs);
    assertThat(options.normalize).isTrue();
  }

  @Test
  public void testNormalizeNoNoBoolean() {
    String[] args = {"--nonormalize"};
    String[] normalizedArgs = AndroidOptionsUtils.normalizeBooleanOptions(options, args);
    assertThat(normalizedArgs[0]).isEqualTo("--normalize=false");
    jc.parse(normalizedArgs);
    assertThat(options.normalize).isFalse();
  }
}
