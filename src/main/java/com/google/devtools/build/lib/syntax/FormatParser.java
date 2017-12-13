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
package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Printer.BasePrinter;
import java.util.List;
import java.util.Map;

/**
 * A helper class that offers a subset of the functionality of Python's string#format.
 *
 * <p> Currently, both manual and automatic positional as well as named replacement
 * fields are supported. However, nested replacement fields are not allowed.
 */
public final class FormatParser {

  private static final ImmutableSet<Character> ILLEGAL_IN_FIELD =
      ImmutableSet.of('.', '[', ']', ',');

  private final Location location;

  public FormatParser(Location location) {
    this.location = location;
  }

  /**
   * Formats the given input string by using the given arguments
   *
   * <p>This method offers a subset of the functionality of Python's string#format
   *
   * @param input The string to be formatted
   * @param args Positional arguments
   * @param kwargs Named arguments
   * @return The formatted string
   */
  public String format(String input, List<Object> args, Map<String, Object> kwargs)
      throws EvalException {
    char[] chars = input.toCharArray();
    StringBuilder output = new StringBuilder();
    History history = new History();

    for (int pos = 0; pos < chars.length; ++pos) {
      char current = chars[pos];
      int advancePos = 0;

      if (current == '{') {
        advancePos = processOpeningBrace(chars, pos, args, kwargs, history, output);
      } else if (current == '}') {
        advancePos = processClosingBrace(chars, pos, output);
      } else {
        output.append(current);
      }

      pos += advancePos;
    }

    return output.toString();
  }

  /**
   * Processes the expression after an opening brace (possibly a replacement field) and emits
   * the result to the output StringBuilder
   *
   * @param chars The entire string
   * @param pos The position of the opening brace
   * @param args List of positional arguments
   * @param kwargs Map of named arguments
   * @param history Helper object that tracks information about previously seen positional
   *    replacement fields
   * @param output StringBuilder that consumes the result
   * @return Number of characters that have been consumed by this method
   */
  protected int processOpeningBrace(
      char[] chars,
      int pos,
      List<Object> args,
      Map<String, Object> kwargs,
      History history,
      StringBuilder output)
      throws EvalException {
    BasePrinter printer = Printer.getPrinter(output);
    if (has(chars, pos + 1, '{')) {
      // Escaped brace -> output and move to char after right brace
      printer.append("{");
      return 1;
    }

    // Inside a replacement field
    String key = getFieldName(chars, pos);
    Object value = null;

    // Only positional replacement fields will lead to a valid index
    try {
      int index = parsePositional(key, history);

      if (index < 0 || index >= args.size()) {
        fail("No replacement found for index " + index);
      }

      value = args.get(index);
    } catch (NumberFormatException nfe) {
      // Non-integer index -> Named
      if (!kwargs.containsKey(key)) {
        fail("Missing argument '" + key + "'");
      }

      value = kwargs.get(key);
    }

    // Format object for output
    printer.str(value);

    // Advances the current position to the index of the closing brace of the
    // replacement field. Due to the definition of the enclosing for() loop,
    // the next iteration will examine the character right after the brace.
    return key.length() + 1;
  }

  /**
   * Processes a closing brace and emits the result to the output StringBuilder
   *
   * @param chars The entire string
   * @param pos Position of the closing brace
   * @param output StringBuilder that consumes the result
   * @return Number of characters that have been consumed by this method
   */
  protected int processClosingBrace(char[] chars, int pos, StringBuilder output)
      throws EvalException {
    if (!has(chars, pos + 1, '}')) {
      // Invalid brace outside replacement field
      fail("Found '}' without matching '{'");
    }

    // Escaped brace -> output and move to char after right brace
    output.append("}");
    return 1;
  }

  /**
   * Checks whether the given input string has a specific character at the given location
   *
   * @param data Input string as character array
   * @param pos Position to be checked
   * @param needle Character to be searched for
   * @return True if string has the specified character at the given location
   */
  protected boolean has(char[] data, int pos, char needle) {
    return pos < data.length && data[pos] == needle;
  }

  /**
   * Extracts the name/index of the replacement field that starts at the specified location
   *
   * @param chars Input string
   * @param openingBrace Position of the opening brace of the replacement field
   * @return Name or index of the current replacement field
   */
  protected String getFieldName(char[] chars, int openingBrace) throws EvalException {
    StringBuilder result = new StringBuilder();
    boolean foundClosingBrace = false;

    for (int pos = openingBrace + 1; pos < chars.length; ++pos) {
      char current = chars[pos];

      if (current == '}') {
        foundClosingBrace = true;
        break;
      } else {
        if (current == '{') {
          fail("Nested replacement fields are not supported");
        } else if (ILLEGAL_IN_FIELD.contains(current)) {
          fail("Invalid character '" + current + "' inside replacement field");
        }

        result.append(current);
      }
    }

    if (!foundClosingBrace) {
      fail("Found '{' without matching '}'");
    }

    return result.toString();
  }

  /**
   * Converts the given key into an integer or assigns the next available index, if empty.
   *
   * @param key Key to be converted
   * @param history Helper object that tracks information about previously seen positional
   *     replacement fields
   * @return The integer equivalent of the key
   */
  protected int parsePositional(String key, History history) throws EvalException {
    int result = -1;

    try {
      if (key.isEmpty()) {
        // Automatic positional -> a new index value has to be assigned
        history.setAutomaticPositional();
        result = history.getNextPosition();
      } else {
        // This will fail if key is a named argument
        result = Integer.parseInt(key);
        history.setManualPositional(); // Only register if the conversion succeeds
      }
    } catch (MixedTypeException mte) {
      fail(mte.getMessage());
    }

    return result;
  }

  /**
   * Throws an exception with the specified error message
   * @param msg The message to be thrown
   */
  protected void fail(String msg) throws EvalException {
    throw new EvalException(location, msg);
  }

  /**
   * Exception for invalid combinations of replacement field types
   */
  private static final class MixedTypeException extends Exception {
    public MixedTypeException() {
      super("Cannot mix manual and automatic numbering of positional fields");
    }
  }

  /**
   * A wrapper to keep track of information about previous replacement fields
   */
  private static final class History {
    /**
     * Different types of positional replacement fields
     */
    private enum Positional {
      NONE,
      MANUAL, // {0}, {1} etc.
      AUTOMATIC // {}
    }

    private Positional type = Positional.NONE;
    private int position = -1;

    /**
     * Returns the next available index for an automatic positional replacement field
     * @return Next index
     */
    public int getNextPosition() {
      ++position;
      return position;
    }

    /**
     * Registers a manual positional replacement field
     */
    public void setManualPositional() throws MixedTypeException {
      setPositional(Positional.MANUAL);
    }

    /**
     * Registers an automatic positional replacement field
     */
    public void setAutomaticPositional() throws MixedTypeException {
      setPositional(Positional.AUTOMATIC);
    }

    /**
     * Indicates that a positional replacement field of the specified type is being
     * processed and checks whether this conflicts with any previously seen
     * replacement fields
     *
     * @param current Type of current replacement field
     */
    protected void setPositional(Positional current) throws MixedTypeException {
      if (type == Positional.NONE) {
        type = current;
      } else if (type != current) {
        throw new MixedTypeException();
      }
    }
  }
}
