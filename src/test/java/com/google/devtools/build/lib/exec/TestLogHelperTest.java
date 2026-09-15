// Copyright 2022 The Bazel Authors. All rights reserved.
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
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.vfs.Path;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for TestLogHelper. */
@RunWith(JUnit4.class)
public final class TestLogHelperTest extends FoundationTestCase {

  @Test
  public void testShouldOutputTestLog() {
    assertThat(TestLogHelper.shouldOutputTestLog(ExecutionOptions.TestOutputFormat.ALL, true))
        .isTrue();
    assertThat(TestLogHelper.shouldOutputTestLog(ExecutionOptions.TestOutputFormat.ALL, false))
        .isTrue();
    assertThat(TestLogHelper.shouldOutputTestLog(ExecutionOptions.TestOutputFormat.ERRORS, false))
        .isTrue();
    assertThat(TestLogHelper.shouldOutputTestLog(ExecutionOptions.TestOutputFormat.ERRORS, true))
        .isFalse();
  }

  @Test
  public void testFormatEmptyTestLog() throws IOException {
    Path logPath = scratch.file("/test.log", "");
    String result = getTestLog(logPath, "testFormatEmptyTestLog");
    assertThat(result)
        .isEqualTo(
            "==================== Test output for testFormatEmptyTestLog:\n"
                + "\n"
                + "================================================================================\n");
  }

  @Test
  public void testFormatTestLogWithHeader() throws IOException {
    Path logPath =
        scratch.file("/test.log", "Header", TestLogHelper.HEADER_DELIMITER, "Empty line");
    String result = getTestLog(logPath, "testFormatTestLogWithHeader");
    assertThat(result)
        .isEqualTo(
            "==================== Test output for testFormatTestLogWithHeader:\n"
                + "Empty line\n"
                + "================================================================================\n");
  }

  @Test
  public void testFormatTestLogWithNoHeader() throws IOException {
    Path logPath = scratch.file("/test.log", "Line 1", "Line 2");
    String result = getTestLog(logPath, "testFormatTestLogWithNoHeader");
    assertThat(result)
        .isEqualTo(
            "==================== Test output for testFormatTestLogWithNoHeader:\n"
                + "Line 1\n"
                + "Line 2\n"
                + "================================================================================\n");
  }

  @Test
  public void testFormatTestLogTooLarge() throws IOException {
    Path logPath = scratch.file("/test.log", new byte[1000]);

    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    TestLogHelper.writeTestLog(logPath, "myTest", bytesOut, /*maxTestOutputBytes=*/ 10);
    assertThat(bytesOut.toString(ISO_8859_1))
        .isEqualTo(
            "==================== Test output for myTest:\n"
                + "Test log too large (1000 > 10), skipping...\n"
                + "================================================================================\n");
  }

  @Test
  public void testFormatTestLog0ByteMax0ByteFilePrintsNothing() throws IOException {
    Path logPath = scratch.file("/test.log", new byte[0]);

    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    TestLogHelper.writeTestLog(logPath, "myTest", bytesOut, /*maxTestOutputBytes=*/ 0);
    assertThat(bytesOut.toString(ISO_8859_1))
        .isEqualTo(
            "==================== Test output for myTest:\n"
                + "================================================================================\n");
  }

  @Test
  public void testFormatTestLogMaxBytesIncludesHeader() throws IOException {
    Path logPath = scratch.file("/test.log", TestLogHelper.HEADER_DELIMITER.getBytes(ISO_8859_1));

    int testLogHeaderLength = TestLogHelper.HEADER_DELIMITER.length();
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    TestLogHelper.writeTestLog(
        logPath, "myTest", bytesOut, /*maxTestOutputBytes=*/ testLogHeaderLength - 1);
    assertThat(bytesOut.toString(ISO_8859_1))
        .isEqualTo(
            "==================== Test output for myTest:\n"
                + String.format(
                    "Test log too large (%d > %d), skipping...\n",
                    testLogHeaderLength, testLogHeaderLength - 1)
                + "================================================================================\n");
  }

  private String getTestLog(Path path, String name) throws IOException {
    ByteArrayOutputStream output = new ByteArrayOutputStream();
    TestLogHelper.writeTestLog(path, name, output, /*maxTestOutputBytes=*/ -1);
    return output.toString(ISO_8859_1);
  }
}
