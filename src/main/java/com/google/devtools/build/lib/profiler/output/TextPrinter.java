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

import com.google.common.base.Preconditions;
import java.io.PrintStream;

/**
 * Utility function for writing text data to a {@link PrintStream}.
 */
public abstract class TextPrinter {

  protected static final String TWO_COLUMN_FORMAT = "%-37s %10s";
  protected static final String THREE_COLUMN_FORMAT = "%-28s %10s %8s";

  private static final String INDENT = "  ";
  protected final PrintStream out;
  private StringBuffer indent;

  protected TextPrinter(PrintStream out) {
    this.out = out;
    this.indent = new StringBuffer();
  }

  /**
   * Increase indentation level
   */
  protected void down() {
    indent.append(INDENT);
  }

  /**
   * Decrease indentation level
   */
  protected void up() {
    Preconditions.checkState(
        indent.length() >= INDENT.length(),
        "Cannot reduce indentation level, this/a previous call to up() is not matched by down().");
    indent.setLength(indent.length() - INDENT.length());
  }

  protected void print(Object text) {
    out.print(text);
  }

  protected void printLn() {
    out.println();
  }

  /** print text and a newline */
  protected void printLn(String text) {
    out.print(text);
    printLn();
  }

  /**
   * newline and indent by current indentation level
   */
  protected void lnIndent() {
    printLn();
    out.print(indent);
  }

  /**
   * newline, indent and print the Object
   * @see PrintStream#print(Object)
   */
  protected void lnPrint(Object text) {
    lnIndent();
    out.print(text);
  }

  /**
   * newline, indent and print the formatted text
   */
  protected void lnPrintf(String format, Object... args) {
    lnIndent();
    out.printf(format, args);
  }

  protected void printf(String format, Object... args) {
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

