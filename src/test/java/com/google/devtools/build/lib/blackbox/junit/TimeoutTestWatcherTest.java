// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.blackbox.junit;

import static com.google.common.truth.Truth.assertThat;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for {@link TimeoutTestWatcher}. Tests both that the timeout was intercepted, and the
 * assertion from the test was not blocked.
 */
@RunWith(JUnit4.class)
public class TimeoutTestWatcherTest extends TimeoutTestWatcherBaseTest {
  @After
  public void tearDown() {
    if ("testTimeoutCaught".equals(testWatcher.getName())) {
      assertThat(timeoutCaught).isTrue();
    }
  }

  /** Test that timeout handler is called */
  @Test
  public void testTimeoutCaught() throws Exception {
    for (int i = 0; i < 10; i++) {
      Thread.sleep(500);
    }
  }

  /** Test that normal test failures are not blocked */
  @Test(expected = AssertionError.class)
  public void testFailure() {
    Assert.fail();
  }
}
