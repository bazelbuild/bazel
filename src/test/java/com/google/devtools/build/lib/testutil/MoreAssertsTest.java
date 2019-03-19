// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertContainsSublist;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertDoesNotContainSublist;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link com.google.devtools.build.lib.testutil.MoreAsserts}. */
@RunWith(JUnit4.class)
public class MoreAssertsTest {

  @Test
  public void testAssertContainsSublistSuccess() {
    List<String> actual = Arrays.asList("a", "b", "c");

    // All single-string combinations.
    assertContainsSublist(actual, "a");
    assertContainsSublist(actual, "b");
    assertContainsSublist(actual, "c");

    // All two-string combinations.
    assertContainsSublist(actual, "a", "b");
    assertContainsSublist(actual, "b", "c");

    // The whole list.
    assertContainsSublist(actual, "a", "b", "c");
  }

  @Test
  public void testAssertContainsSublistFailure() {
    List<String> actual = Arrays.asList("a", "b", "c");

    try {
      assertContainsSublist(actual, "d");
      fail("no exception thrown");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().startsWith("Did not find [d] as a sublist of [a, b, c]");
    }

    try {
      assertContainsSublist(actual, "a", "c");
      fail("no exception thrown");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().startsWith("Did not find [a, c] as a sublist of [a, b, c]");
    }

    try {
      assertContainsSublist(actual, "b", "c", "d");
      fail("no exception thrown");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().startsWith("Did not find [b, c, d] as a sublist of [a, b, c]");
    }
  }

  @Test
  public void testAssertDoesNotContainSublistSuccess() {
    List<String> actual = Arrays.asList("a", "b", "c");
    assertDoesNotContainSublist(actual, "d");
    assertDoesNotContainSublist(actual, "a", "c");
    assertDoesNotContainSublist(actual, "b", "c", "d");
  }

  @Test
  public void testAssertDoesNotContainSublistFailure() {
    List<String> actual = Arrays.asList("a", "b", "c");

    // All single-string combinations.
    try {
      assertDoesNotContainSublist(actual, "a");
      fail("no exception thrown");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().isEqualTo("Found [a] as a sublist of [a, b, c]");
    }
    try {
      assertDoesNotContainSublist(actual, "b");
      fail("no exception thrown");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().isEqualTo("Found [b] as a sublist of [a, b, c]");
    }
    try {
      assertDoesNotContainSublist(actual, "c");
      fail("no exception thrown");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().isEqualTo("Found [c] as a sublist of [a, b, c]");
    }

    // All two-string combinations.
    try {
      assertDoesNotContainSublist(actual, "a", "b");
      fail("no exception thrown");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().isEqualTo("Found [a, b] as a sublist of [a, b, c]");
    }
    try {
      assertDoesNotContainSublist(actual, "b", "c");
      fail("no exception thrown");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().isEqualTo("Found [b, c] as a sublist of [a, b, c]");
    }

    // The whole list.
    try {
      assertDoesNotContainSublist(actual, "a", "b", "c");
      fail("no exception thrown");
    } catch (AssertionError e) {
      assertThat(e).hasMessageThat().isEqualTo("Found [a, b, c] as a sublist of [a, b, c]");
    }
  }

}
