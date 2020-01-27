// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.windows;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link WindowsShortPath}. */
@RunWith(JUnit4.class)
public class WindowsShortPathTest {
  @Test
  public void testShortNameMatcher() {
    assertThat(WindowsShortPath.isShortPath("abc")).isFalse(); // no ~ in the name
    assertThat(WindowsShortPath.isShortPath("abc~")).isFalse(); // no number after the ~
    assertThat(WindowsShortPath.isShortPath("~abc")).isFalse(); // no ~ followed by number
    assertThat(WindowsShortPath.isShortPath("too_long_path")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("too_long_path~1")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("abcd~1234")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("h~1")).isTrue();
    assertThat(WindowsShortPath.isShortPath("h~12")).isTrue();
    assertThat(WindowsShortPath.isShortPath("h~12.")).isTrue();
    assertThat(WindowsShortPath.isShortPath("h~12.a")).isTrue();
    assertThat(WindowsShortPath.isShortPath("h~12.abc")).isTrue();
    assertThat(WindowsShortPath.isShortPath("h~123456")).isTrue();
    assertThat(WindowsShortPath.isShortPath("hellow~1")).isTrue();
    assertThat(WindowsShortPath.isShortPath("hellow~1.")).isTrue();
    assertThat(WindowsShortPath.isShortPath("hellow~1.a")).isTrue();
    assertThat(WindowsShortPath.isShortPath("hellow~1.abc")).isTrue();
    assertThat(WindowsShortPath.isShortPath("hello~1.abcd")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("hellow~1.abcd")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("hello~12")).isTrue();
    assertThat(WindowsShortPath.isShortPath("hello~12.")).isTrue();
    assertThat(WindowsShortPath.isShortPath("hello~12.a")).isTrue();
    assertThat(WindowsShortPath.isShortPath("hello~12.abc")).isTrue();
    assertThat(WindowsShortPath.isShortPath("hello~12.abcd")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("hellow~12")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("hellow~12.")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("hellow~12.a")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("hellow~12.ab")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("~h~1")).isTrue();
    assertThat(WindowsShortPath.isShortPath("~h~1.")).isTrue();
    assertThat(WindowsShortPath.isShortPath("~h~1.a")).isTrue();
    assertThat(WindowsShortPath.isShortPath("~h~1.abc")).isTrue();
    assertThat(WindowsShortPath.isShortPath("~h~1.abcd")).isFalse(); // too long for 8dot3
    assertThat(WindowsShortPath.isShortPath("~h~12")).isTrue();
    assertThat(WindowsShortPath.isShortPath("~h~12~1")).isTrue();
    assertThat(WindowsShortPath.isShortPath("~h~12~1.")).isTrue();
    assertThat(WindowsShortPath.isShortPath("~h~12~1.a")).isTrue();
    assertThat(WindowsShortPath.isShortPath("~h~12~1.abc")).isTrue();
    assertThat(WindowsShortPath.isShortPath("~h~12~1.abcd")).isFalse(); // too long for 8dot3
  }
}
