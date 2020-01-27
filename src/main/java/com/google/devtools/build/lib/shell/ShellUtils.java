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

package com.google.devtools.build.lib.shell;

import java.util.List;

/**
 * Utility functions for Bourne shell commands, including escaping and
 * tokenizing.
 */
public abstract class ShellUtils {

  private ShellUtils() {}

  /**
   * Characters that have no special meaning to the shell.
   */
  private static final String SAFE_PUNCTUATION = "@%-_+:,./";

  /**
   * Quotes a word so that it can be used, without further quoting,
   * as an argument (or part of an argument) in a shell command.
   */
  public static String shellEscape(String word) {
    int len = word.length();
    if (len == 0) {
      // Empty string is a special case: needs to be quoted to ensure that it gets
      // treated as a separate argument.
      return "''";
    }
    for (int ii = 0; ii < len; ii++) {
      char c = word.charAt(ii);
      // We do this positively so as to be sure we don't inadvertently forget
      // any unsafe characters.
      if (!Character.isLetterOrDigit(c) && SAFE_PUNCTUATION.indexOf(c) == -1) {
        // replace() actually means "replace all".
        return "'" + word.replace("'", "'\\''") + "'";
      }
    }
    return word;
  }

  /**
   * Given an argv array such as might be passed to execve(2), returns a string
   * that can be copied and pasted into a Bourne shell for a similar effect.
   */
  public static String prettyPrintArgv(List<String> argv) {
    StringBuilder buf = new StringBuilder();
    for (String arg: argv) {
      if (buf.length() > 0) {
        buf.append(' ');
      }
      buf.append(shellEscape(arg));
    }
    return buf.toString();
  }


  /**
   * Thrown by tokenize method if there is an error
   */
  public static class TokenizationException extends Exception {
    TokenizationException(String message) {
      super(message);
    }
  }

  /**
   * Populates the passed list of command-line options extracted from {@code
   * optionString}, which is a string containing multiple options, delimited in
   * a Bourne shell-like manner.
   *
   * @param options the list to be populated with tokens.
   * @param optionString the string to be tokenized.
   * @throws TokenizationException if there was an error (such as an
   * unterminated quotation).
   */
  public static void tokenize(List<String> options, String optionString)
      throws TokenizationException {
    // See test suite for examples.
    //
    // Note: backslash escapes the following character, except within a
    // single-quoted region where it is literal.

    StringBuilder token = new StringBuilder();
    boolean forceToken = false;
    char quotation = '\0'; // NUL, '\'' or '"'
    for (int ii = 0, len = optionString.length(); ii < len; ii++) {
      char c = optionString.charAt(ii);
      if (quotation != '\0') { // in quotation
        if (c == quotation) { // end of quotation
          quotation = '\0';
        } else if (c == '\\' && quotation == '"') { // backslash in "-quotation
          if (++ii == len) {
            throw new TokenizationException("backslash at end of string");
          }
          c = optionString.charAt(ii);
          if (c != '\\' && c != '"') {
            token.append('\\');
          }
          token.append(c);
        } else { // regular char, in quotation
          token.append(c);
        }
      } else { // not in quotation
        if (c == '\'' || c == '"') { // begin single/double quotation
          quotation = c;
          forceToken = true;
        } else if (c == ' ' || c == '\t') { // space, not quoted
          if (forceToken || token.length() > 0) {
            options.add(token.toString());
            token = new StringBuilder();
            forceToken = false;
          }
        } else if (c == '\\') { // backslash, not quoted
          if (++ii == len) {
            throw new TokenizationException("backslash at end of string");
          }
          token.append(optionString.charAt(ii));
        } else { // regular char, not quoted
          token.append(c);
        }
      }
    }
    if (quotation != '\0') {
      throw new TokenizationException("unterminated quotation");
    }
    if (forceToken || token.length() > 0) {
      options.add(token.toString());
    }
  }

  /**
   * Escape command line arguments for {@code CreateProcessW} on Windows.
   *
   * <p>This method implements the same algorithm as the native xx_binary launcher does (see
   * https://github.com/bazelbuild/bazel/pull/7411).
   *
   * <p>A similar algorithm with lots of background information is described here:
   * https://blogs.msdn.microsoft.com/twistylittlepassagesallalike/2011/04/23/
   */
  public static String windowsEscapeArg(String s) {
    if (s.isEmpty()) {
      return "\"\"";
    } else {
      boolean needsEscape = false;
      for (int i = 0; i < s.length(); ++i) {
        char c = s.charAt(i);
        if (c == ' ' || c == '"') {
          needsEscape = true;
          break;
        }
      }
      if (!needsEscape) {
        return s;
      }
    }

    StringBuilder result = new StringBuilder();
    result.append('"');
    int start = 0;
    for (int i = 0; i < s.length(); ++i) {
      char c = s.charAt(i);
      if (c == '"' || c == '\\') {
        // Copy the segment since the last special character.
        if (start >= 0) {
          result.append(s, start, i);
          start = -1;
        }

        // Handle the current special character.
        if (c == '"') {
          // This is a quote character. Escape it with a single backslash.
          result.append("\\\"");
        } else {
          // This is a backslash (or the first one in a run of backslashes).
          // Whether we escape it depends on whether the run ends with a quote.
          int runLen = 1;
          int j = i + 1;
          while (j < s.length() && s.charAt(j) == '\\') {
            runLen++;
            j++;
          }
          if (j == s.length()) {
            // The run of backslashes goes to the end.
            // We have to escape every backslash with another backslash.
            for (int k = 0; k < runLen * 2; ++k) {
              result.append('\\');
            }
            break;
          } else if (j < s.length() && s.charAt(j) == '"') {
            // The run of backslashes is terminated by a quote.
            // We have to escape every backslash with another backslash, and
            // escape the quote with one backslash.
            for (int k = 0; k < runLen * 2; ++k) {
              result.append('\\');
            }
            result.append("\\\"");
            i += runLen; // 'i' is also increased in the loop iteration step
          } else {
            // No quote found. Each backslash counts for itself, they must not be
            // escaped.
            for (int k = 0; k < runLen; ++k) {
              result.append('\\');
            }
            i += runLen - 1; // 'i' is also increased in the loop iteration step
          }
        }
      } else {
        // This is not a special character. Start the segment if necessary.
        if (start < 0) {
          start = i;
        }
      }
    }
    // Save final segment after the last special character.
    if (start != -1) {
      result.append(s, start, s.length());
    }
    result.append('"');
    return result.toString();
  }
}
