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

import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.exec.TestStrategy.TestOutputFormat;
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

  public static final String HEADER_DELIMITER =
      "-----------------------------------------------------------------------------";

  /**
   * Determines whether the test log should be output from the current outputMode and whether the
   * test has passed or not.
   */
  public static boolean shouldOutputTestLog(TestOutputFormat outputMode, boolean hasPassed) {
    return (outputMode == TestOutputFormat.ALL)
        || (!hasPassed && (outputMode == TestOutputFormat.ERRORS));
  }

  /**
   * Reads the contents of the test log from the provided testOutput file, adds header and footer
   * and returns the result. This method also looks for a header delimiter and cuts off the text
   * before it, except if the header is 50 lines or longer.
   */
  public static void writeTestLog(Path testOutput, String testName, OutputStream out)
      throws IOException {
    PrintStream printOut = new PrintStream(new BufferedOutputStream(out));
    try {
      final String outputHeader = "==================== Test output for " + testName + ":";
      final String outputFooter =
          "================================================================================";

      printOut.println(outputHeader);
      printOut.flush();

      FilterTestHeaderOutputStream filteringOutputStream = getHeaderFilteringOutputStream(printOut);
      try (InputStream input = testOutput.getInputStream()) {
        ByteStreams.copy(input, filteringOutputStream);
      }

      if (!filteringOutputStream.foundHeader()) {
        try (InputStream inputAgain = testOutput.getInputStream()) {
          ByteStreams.copy(inputAgain, out);
        }
      }

      printOut.println(outputFooter);
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
}
