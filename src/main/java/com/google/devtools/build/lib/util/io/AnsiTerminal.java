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

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * A class which encapsulates the fancy curses-type stuff that you can do using
 * standard ANSI terminal control sequences.
 */
public class AnsiTerminal {
  /**
   * An enumeration of all terminal colors, containing the escape sequences for both background and
   * foreground settings.
   */
  public enum Color {
    RED("^[31m", "^[41m"),
    GREEN("^[32m", "^[42m"),
    YELLOW("^[33m", "^[43m"),
    BLUE("^[34m", "^[44m"),
    MAGENTA("^[35m", "^[45m"),
    CYAN("^[36m", "^[46m"),
    GRAY("^[37m", "^[47m"),

    DEFAULT("^[0m", "^[0m");

    private final byte[] escapeSeq;
    private final byte[] backgroundEscapeSeq;

    private Color(String escapeSeq, String backgroundEscapeSeq) {
      this.escapeSeq = escapeSeq.replace('^', (char) 27).getBytes(StandardCharsets.US_ASCII);
      this.backgroundEscapeSeq =
          backgroundEscapeSeq.replace('^', (char) 27).getBytes(StandardCharsets.US_ASCII);
    }
  }

  private static final byte[] ESC = {27, (byte) '['};
  private static final byte BEL = 7;
  private static final byte UP = (byte) 'A';
  private static final byte ERASE_LINE = (byte) 'K';
  private static final byte SET_GRAPHICS = (byte) 'm';
  private static final byte TEXT_BOLD = (byte) '1';
  private static final byte[] SET_TERM_TITLE = {27, (byte) ']', (byte) '0', (byte) ';'};

  public static byte[] CR = { 13 };

  private final OutputStream out;

  /**
   * Creates an AnsiTerminal object wrapping an output stream which is going to
   * be displayed in an ANSI compatible terminal or shell window.
   *
   * @param out the output stream
   */
  public AnsiTerminal(OutputStream out) {
    this.out = out;
  }

  /**
   * Moves the cursor upwards by a specified number of lines. This will not
   * cause any scrolling if it tries to move above the top of the terminal
   * window.
   */
  public void cursorUp(int numLines) throws IOException {
    writeBytes(ESC, ("" + numLines).getBytes(), new byte[] { UP });
  }

  /**
   * Clear the current terminal line from the cursor position to the end.
   */
  public void clearLine() throws IOException {
    writeEscapeSequence(ERASE_LINE);
  }

  /**
   * Makes any text output to the terminal appear in bold.
   */
  public void textBold() throws IOException {
    writeEscapeSequence(TEXT_BOLD,  SET_GRAPHICS);
  }

  /**
   * Set the color of the foreground or background of the terminal.
   *
   * @param color one of the foreground or background color constants
   */
  public void setTextColor(Color color) throws IOException {
    writeBytes(color.escapeSeq);
  }

  /**
   * Set the color of the foreground or background of the terminal.
   *
   * @param color one of the foreground or background color constants
   */
  public void setBackgroundColor(Color color) throws IOException {
    writeBytes(color.backgroundEscapeSeq);
  }

  /**
   * Resets the terminal colors and fonts to defaults.
   */
  public void resetTerminal() throws IOException {
    writeEscapeSequence((byte)'0', (byte)'m');
  }

  /**
   * Makes text print on the terminal in red.
   */
  public void textRed() throws IOException {
    setTextColor(Color.RED);
  }

  /**
   * Makes text print on the terminal in blue.
   */
  public void textBlue() throws IOException {
    setTextColor(Color.BLUE);
  }

  /**
   * Makes text print on the terminal in green.
   */
  public void textGreen() throws IOException {
    setTextColor(Color.GREEN);
  }

  /**
   * Makes text print on the terminal in magenta.
   */
  public void textMagenta() throws IOException {
    setTextColor(Color.MAGENTA);
  }

  /**
   * Makes text print on the terminal in yellow.
   */
  public void textYellow() throws IOException {
    setTextColor(Color.YELLOW);
  }

  /**
   * Set the terminal title.
   */
  public void setTitle(String title) throws IOException {
    writeBytes(SET_TERM_TITLE, title.getBytes(), new byte[] { BEL });
  }

  /**
   * Writes a string to the terminal using the current font, color and cursor
   * position settings.
   *
   * @param text the text to write
   */
  public void writeString(String text) throws IOException {
    out.write(text.getBytes());
  }

  /**
   * Writes a byte sequence to the terminal using the current font, color and cursor position
   * settings.
   *
   * @param bytes the bytes to write
   */
  public void writeBytes(byte[] bytes) throws IOException {
    out.write(bytes);
  }

  /**
   * Utility method for generating control sequences. Takes a collection of byte arrays, which
   * contain the components of a control sequence, concatenates them, and prints them to the
   * terminal.
   *
   * @param stuff the byte arrays that make up the sequence to be sent to the terminal
   */
  private void writeBytes(byte[]... stuff) throws IOException {
    for (byte[] bytes : stuff) {
      out.write(bytes);
    }
  }

  /**
   * Utility method which makes it easier to generate the control sequences for the terminal.
   *
   * @param bytes bytes which should be prefixed with the terminal escape sequence to produce a
   *     valid control sequence
   */
  private void writeEscapeSequence(byte... bytes) throws IOException {
    writeBytes(ESC, bytes);
  }

  /** Sends a carriage return to the terminal. */
  public void cr() throws IOException {
    writeBytes(CR);
  }

  /**
   * Flushes the underlying stream. This class does not do any buffering of its own, but the
   * underlying OutputStream may.
   */
  public void flush() throws IOException {
    out.flush();
  }
}
