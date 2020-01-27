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
package com.google.devtools.build.lib.analysis.test;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.packages.TestTimeout;
import com.google.devtools.build.lib.packages.TestTimeout.TestTimeoutConverter;
import com.google.devtools.common.options.OptionsParsingException;
import java.time.Duration;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link TestTimeoutConverter}.
 */
@RunWith(JUnit4.class)
public class TestTimeoutConverterTest {
  private Map<TestTimeout, Duration> timeouts;

  protected void setTimeouts(String option) throws OptionsParsingException {
    timeouts = new TestTimeoutConverter().convert(option);
  }

  protected void assertTimeout(TestTimeout timeout, int expected) {
    assertThat(timeouts).containsEntry(timeout, Duration.ofSeconds(expected));
  }

  protected void assertDefaultTimeout(TestTimeout timeout) {
    assertTimeout(timeout, timeout.getTimeoutSeconds());
  }

  protected void assertFailure(String option) {
    assertThrows(
        "Incorrectly parsed '" + option + "'",
        OptionsParsingException.class,
        () -> setTimeouts(option));
  }

  @Test
  public void testDefaultTimeout() throws Exception {
    setTimeouts("-1");
    assertDefaultTimeout(TestTimeout.SHORT);
    assertDefaultTimeout(TestTimeout.MODERATE);
    assertDefaultTimeout(TestTimeout.LONG);
    assertDefaultTimeout(TestTimeout.ETERNAL);
  }

  @Test
  public void testUniversalTimeout() throws Exception {
    setTimeouts("1");
    assertTimeout(TestTimeout.SHORT, 1);
    assertTimeout(TestTimeout.MODERATE, 1);
    assertTimeout(TestTimeout.LONG, 1);
    assertTimeout(TestTimeout.ETERNAL, 1);

    setTimeouts("2,"); // comma at the end is ignored.
    assertTimeout(TestTimeout.SHORT, 2);
    assertTimeout(TestTimeout.MODERATE, 2);
    assertTimeout(TestTimeout.LONG, 2);
    assertTimeout(TestTimeout.ETERNAL, 2);
  }

  @Test
  public void testSeparateTimeouts() throws Exception {
    setTimeouts("1,0,-1,3");
    assertTimeout(TestTimeout.SHORT, 1);
    assertDefaultTimeout(TestTimeout.MODERATE);
    assertDefaultTimeout(TestTimeout.LONG);
    assertTimeout(TestTimeout.ETERNAL, 3);

    setTimeouts("0,-1,3,20");
    assertDefaultTimeout(TestTimeout.SHORT);
    assertDefaultTimeout(TestTimeout.MODERATE);
    assertTimeout(TestTimeout.LONG, 3);
    assertTimeout(TestTimeout.ETERNAL, 20);
  }

  @Test
  public void testIncorrectStrings() throws Exception {
    assertFailure("");
    assertFailure("1a");
    assertFailure("1 2 3 4");
    assertFailure("1:2:3:4");
    assertFailure("1,2,3");
    assertFailure("1,2,3,4,");
    assertFailure("1,2,,3,4");
    assertFailure("1,2,3 4");
    assertFailure("1,2,3,4,5");
  }
}
