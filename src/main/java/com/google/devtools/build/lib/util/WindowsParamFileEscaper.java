// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;

/** Utility class to escape strings for use in param files for windows lld-link. */
public final class WindowsParamFileEscaper {
  private WindowsParamFileEscaper() {}

  /**
   * Escapes the @argument to be suitable for lld-link. Existing double-quotes are escaped, and
   * empty arguments and arguments that contain whitespace are surrounded in unescaped
   * double-quotes. Backslashes immediately before a double-quote are doubled.
   *
   * @see <a
   *     href="https://github.com/llvm/llvm-project/blob/4bc3b3501ff994fb3504ed2b973342821a9c8cea/llvm/lib/Support/CommandLine.cpp#L916">LLVM
   *     Parser Implementation</a>
   */
  public static String escapeString(String argument) {
    if (!argument.isEmpty() && argument.indexOf('\\') < 0 && argument.indexOf('"') < 0) {
      return containsWhitespace(argument) ? "\"" + argument + "\"" : argument;
    }
    boolean needsSurroundingQuotes = argument.isEmpty() || containsWhitespace(argument);
    StringBuilder out = new StringBuilder(argument.length() + 2);
    if (needsSurroundingQuotes) {
      out.append("\"");
    }
    int backslashes = 0;
    for (int i = 0; i < argument.length(); i++) {
      char c = argument.charAt(i);
      if (c == '\\') {
        backslashes++;
        continue;
      }
      if (c == '"') {
        // Each literal backslash needs a pair, followed by one to escape the quote.
        out.append("\\".repeat(2 * backslashes + 1));
      } else {
        out.append("\\".repeat(backslashes));
      }
      out.append(c);
      backslashes = 0;
    }
    if (needsSurroundingQuotes) {
      // Keep trailing backslashes from escaping the closing quote.
      out.append("\\".repeat(2 * backslashes));
      out.append("\"");
    } else {
      out.append("\\".repeat(backslashes));
    }
    return out.toString();
  }

  private static final ImmutableList<CharSequence> WHITESPACE_CHARACTERS =
      ImmutableList.of(" ", "\t", "\n", "\r");

  private static boolean containsWhitespace(String argument) {
    return WHITESPACE_CHARACTERS.stream().anyMatch(argument::contains);
  }

  /** Escapes each argument in @unescaped using WindowsParamFileEscaper::escapeString. */
  public static Iterable<String> escapeAll(Iterable<? extends String> unescaped) {
    return Iterables.transform(unescaped, WindowsParamFileEscaper::escapeString);
  }
}
