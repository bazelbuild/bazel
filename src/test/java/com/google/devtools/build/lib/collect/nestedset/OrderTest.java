// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link com.google.devtools.build.lib.collect.nestedset.Order}.
 */
@RunWith(JUnit4.class)
public class OrderTest {

  @Test
  public void testParsing() throws Exception {
    for (Order current : Order.values()) {
      assertThat(Order.parse(current.getStarlarkName())).isEqualTo(current);
    }
  }

  @Test
  public void testForErrors() throws Exception {
    causeError(null);
    causeError("");
    causeError("lol");
    causeError("naive");
    causeError("naivelink");
  }

  private void causeError(String invalidName) throws Exception {
    IllegalArgumentException ex =
        assertThrows(IllegalArgumentException.class, () -> Order.parse(invalidName));
    assertThat(ex).hasMessageThat().startsWith("Invalid order");
  }
}
