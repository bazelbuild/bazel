// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.SingleStringArgFormatter.format;
import static com.google.devtools.build.lib.actions.SingleStringArgFormatter.isValid;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SingleStringArgFormatter} */
@RunWith(JUnit4.class)
public class SingleStringArgFormatterTest {

  @Test
  public void testValidate() {
    assertThat(isValid("hello %s")).isTrue();
    assertThat(isValid("%s hello")).isTrue();
    assertThat(isValid("hello %%s")).isFalse();
    assertThat(isValid("hello %%s %s")).isTrue();
    assertThat(isValid("hello %s %%s")).isTrue();
    assertThat(isValid("hello %s %s")).isFalse();
    assertThat(isValid("%s hello %s")).isFalse();
    assertThat(isValid("hello %")).isFalse();
    assertThat(isValid("hello %f")).isFalse();
    assertThat(isValid("hello %s %f")).isFalse();
    assertThat(isValid("hello %s %")).isFalse();
  }

  @Test
  public void testFormat() {
    assertThat(format("hello %s", "world")).isEqualTo("hello world");
    assertThat(format("%s hello", "world")).isEqualTo("world hello");
    assertThat(format("hello %s, hello", "world")).isEqualTo("hello world, hello");
    assertThat(format("hello %s, hello", "world")).isEqualTo("hello world, hello");
    assertThat(format("hello %%s %s", "world")).isEqualTo("hello %%s world");
    assertThrows(IllegalArgumentException.class, () -> format("hello", "hello"));
    assertThrows(IllegalArgumentException.class, () -> format("hello %s %s", "hello"));
  }
}
