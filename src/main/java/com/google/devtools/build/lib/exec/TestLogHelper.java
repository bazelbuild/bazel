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
package com.google.devtools.build.lib.exec;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.exec.ExecutionOptions.TestOutputFormat;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.Path;
import java.io.BufferedOutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;

/**
 * A helper class for test log handling. It determines whether the test log should be output and
 * formats the test log for console display.
 */
public class TestLogHelper {

  @VisibleForTesting
  static final String HEADER_DELIMITER =
      "-----------------------------------------------------------------------------";

  /**
   * Determines whether the test log should be output from the current outputMode and whether the
   * test has passed or not.
   */
  public static boolean shouldOutputTestLog(TestOutputFormat outputMode, boolean hasPassed) {
    return (outputMode == ExecutionOptions.TestOutputFormat.ALL)
        || (!hasPassed && (outputMode == ExecutionOptions.TestOutputFormat.ERRORS));
  }

  /**
   * Streams the contents of testOutput file to the provided output, adding a new header and footer.
   * The internal test header is elided. The test output is not emitted if its size is greater than
   * the provided threshold.
   *
   * @param maxTestOutputBytes Maximum test log size, including header, to output. Negative implies
   *     no limit.
   */
  public static void writeTestLog(
      Path testOutput, String testName, OutputStream out, int maxTestOutputBytes)
      throws IOException {
    PrintStream printOut = new PrintStream(new BufferedOutputStream(out));
    try {
      printOut.println("==================== Test output for " + testName + ":");
      printOut.flush();

      if (maxTestOutputBytes < 0) {
        // No limit, print it all.
        streamTestLog(testOutput, printOut);
      } else {
        long testOutputBytes = testOutput.getFileSize();
        if (testOutputBytes <= maxTestOutputBytes) {
          streamTestLog(testOutput, printOut);
        } else {
          printOut.printf(
              "Test log too large (%s > %s), skipping...\n", testOutputBytes, maxTestOutputBytes);
        }
      }

      printOut.println(
          "================================================================================");
    } finally {
      printOut.flush();
    }
  }

  /**
   * Returns an output stream that doesn't write to original until it sees HEADER_DELIMITER by
   * itself on a line.
   */
  public static FilterTestHeaderOutputStream getHeaderFilteringOutputStream(OutputStream original) {
    return new FilterTestHeaderOutputStream(original);
  }

  private TestLogHelper() {
    // Prevent Java from creating a public constructor.
  }

  /** Use this class to filter the streaming output of a test until we see the header delimiter. */
  public static class FilterTestHeaderOutputStream extends FilterOutputStream {

    private boolean seenDelimiter = false;
    private StringBuilder lineBuilder = new StringBuilder();

    private static final int NEWLINE = '\n';

    public FilterTestHeaderOutputStream(OutputStream out) {
      super(out);
    }

    @Override
    public void write(int b) throws IOException {
      if (seenDelimiter) {
        out.write(b);
      } else if (b == NEWLINE) {
        String line = lineBuilder.toString();
        lineBuilder = new StringBuilder();
        if (line.equals(TestLogHelper.HEADER_DELIMITER)
            ||
            // On Windows, the line break could be \r\n, we want this case to work as well.
            (OS.getCurrent() == OS.WINDOWS && line.equals(TestLogHelper.HEADER_DELIMITER + "\r"))) {
          seenDelimiter = true;
        }
      } else if (lineBuilder.length() <= TestLogHelper.HEADER_DELIMITER.length()) {
        lineBuilder.append((char) b);
      }
    }

    @Override
    public void write(byte b[], int off, int len) throws IOException {
      if (seenDelimiter) {
        out.write(b, off, len);
      } else {
        super.write(b, off, len);
      }
    }

    public boolean foundHeader() {
      return seenDelimiter;
    }
  }

  private static void streamTestLog(Path fromPath, PrintStream out) throws IOException {
    FilterTestHeaderOutputStream filteringOutputStream = getHeaderFilteringOutputStream(out);
    try (InputStream input = fromPath.getInputStream()) {
      ByteStreams.copy(input, filteringOutputStream);
    }

    if (!filteringOutputStream.foundHeader()) {
      try (InputStream inputAgain = fromPath.getInputStream()) {
        ByteStreams.copy(inputAgain, out);
      }
    }
  }
}
