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

package com.google.devtools.build.lib.util.io;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.EnumSet;
import java.util.logging.Logger;
import java.util.regex.Pattern;

/**
 * Allows to print "colored" strings by parsing predefined string keywords,
 * which, depending on the useColor value are either replaced with ANSI terminal
 * coloring sequences (as defined by the {@link AnsiTerminal} class) or stripped.
 *
 * Supported keywords are defined by the enum {@link AnsiTerminalPrinter.Mode}.
 * Following keywords are supported:
 *   INFO  - switches color to green.
 *   ERROR - switches color to bold red.
 *   WARNING - switches color to magenta.
 *   NORMAL - resets terminal to the default state.
 *
 * Each keyword is starts with prefix "{#" followed by the enum constant name
 * and suffix "#}". Keywords should not be inserted manually - provided enum
 * constants should be used instead.
 */
@ThreadCompatible
public class AnsiTerminalPrinter {

  private static final String MODE_PREFIX = "{#";
  private static final String MODE_SUFFIX = "#}";

  // Mode pattern must match MODE_PREFIX and do lookahead for the rest of the
  // mode string.
  private static final String MODE_PATTERN = "\\{\\#(?=[A-Z]+\\#\\})";

  /**
   * List of supported coloring modes for the {@link AnsiTerminalPrinter}.
   */
  public static enum Mode {
    INFO,     // green
    ERROR,    // bold red
    WARNING,  // magenta
    DEFAULT;  // default color

    @Override
    public String toString() {
      return MODE_PREFIX + name() + MODE_SUFFIX;
    }
  }

  private static final Logger logger = Logger.getLogger(AnsiTerminalPrinter.class.getName());
  private static final EnumSet<Mode> MODES = EnumSet.allOf(Mode.class);
  private static final Pattern PATTERN = Pattern.compile(MODE_PATTERN);

  private final OutputStream stream;
  private final PrintWriter writer;
  private final AnsiTerminal terminal;
  private boolean useColor;
  private Mode lastMode = Mode.DEFAULT;

  /**
   * Creates new instance using provided OutputStream and sets coloring logic
   * for that instance.
   */
  public AnsiTerminalPrinter(OutputStream out, boolean useColor) {
    this.useColor = useColor;
    terminal = new AnsiTerminal(out);
    writer = new PrintWriter(out, true);
    stream = out;
  }

  /**
   * Writes the specified string to the output stream while injecting coloring
   * sequences when appropriate mode keyword is found and flushes.
   *
   * List of supported mode keywords is defined by the enum {@link Mode}.
   *
   * See class documentation for details.
   */
  public void print(String str) {
    for (String part : PATTERN.split(str)) {
      int index = part.indexOf(MODE_SUFFIX);
      // Mode name will contain at least one character, so suffix index
      // must be at least 1. If it isn't then there is no match.
      if (index > 1) {
        for (Mode mode : MODES) {
          if (index == mode.name().length() && part.startsWith(mode.name())) {
            setupTerminal(mode);
            part = part.substring(index + MODE_SUFFIX.length());
            break;
          }
        }
      }
      writer.print(part);
      writer.flush();
    }
  }

  public void printLn(String str) {
    print(str + "\n");
  }

  /**
   * Returns the underlying OutputStream.
   */
  public OutputStream getOutputStream() {
    return stream;
  }

  /**
   * Injects coloring escape sequences if output should be colored and mode
   * has been changed.
   */
  private void setupTerminal(Mode mode) {
    if (!useColor) {
      return;
    }
    try {
      if (lastMode != mode) {
        terminal.resetTerminal();
        lastMode = mode;
        if (mode == Mode.DEFAULT) {
          return; // Terminal is already reset - nothing else to do.
        } else if (mode == Mode.INFO) {
          terminal.textGreen();
        } else if (mode == Mode.ERROR) {
          terminal.textRed();
          terminal.textBold();
        } else if (mode == Mode.WARNING) {
          terminal.textMagenta();
        }
      }
    } catch (IOException e) {
      // AnsiTerminal state is now considered to be inconsistent - coloring
      // should be disabled to prevent future use of AnsiTerminal instance.
      logger.warning("Disabling coloring due to " + e.getMessage());
      useColor = false;
    }
  }
}
