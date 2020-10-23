// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler.output;

import java.io.PrintStream;

/**
 * Utility function for writing text data to a {@link PrintStream}.
 */
public abstract class TextPrinter {

  protected static final String THREE_COLUMN_FORMAT = "%-28s %10s %8s";

  protected final PrintStream out;

  protected TextPrinter(PrintStream out) {
    this.out = out;
  }

  protected void printLn() {
    out.println();
  }

  /** newline and print the Object */
  protected void lnPrint(Object text) {
    out.println();
    out.print(text);
  }

  /** newline and print the formatted text */
  protected void lnPrintf(String format, Object... args) {
    out.println();
    out.printf(format, args);
  }

  /**
   * Represents a double value as either "N/A" if it is NaN, or as a percentage with "%.2f%%".
   * @param relativeValue is assumed to be a ratio of two values and will be multiplied with 100
   *    for output
   */
  public static String prettyPercentage(double relativeValue) {
    if (Double.isNaN(relativeValue)) {
      return "N/A";
    }
    return String.format("%.2f%%", relativeValue * 100);
  }
}

