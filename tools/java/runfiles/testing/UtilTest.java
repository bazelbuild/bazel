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

package com.google.devtools.build.runfiles;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link Util}. */
@RunWith(JUnit4.class)
public final class UtilTest {

  @Test
  public void testIsNullOrEmpty() {
    assertThat(Util.isNullOrEmpty(null)).isTrue();
    assertThat(Util.isNullOrEmpty("")).isTrue();
    assertThat(Util.isNullOrEmpty("\0")).isFalse();
    assertThat(Util.isNullOrEmpty("some text")).isFalse();
  }

  @Test
  public void testCheckArgument() {
    Util.checkArgument(true, null, null);

    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> Util.checkArgument(false, null, null));
    assertThat(e).hasMessageThat().isEqualTo("argument validation failed");

    e = assertThrows(IllegalArgumentException.class, () -> Util.checkArgument(false, "foo-%s", 42));
    assertThat(e).hasMessageThat().isEqualTo("foo-42");
  }
}
