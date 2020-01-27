// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.collect.Iterables;
import com.google.common.escape.CharEscaper;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Utility class to escape strings for use in param files for gcc or clang.
 *
 * <p>Gcc and Clang interpret the following characters specially: single quote ('), double quote
 * ("), backslash (\), space ( ), tab (\t), carriage return (\r), newline (\n), form feed (\f), and
 * vertical tab (\u000B). All can be escaped by prefixing the symbol with a backslash.
 */
@Immutable
public final class GccParamFileEscaper extends CharEscaper {
  public static final GccParamFileEscaper INSTANCE = new GccParamFileEscaper();

  private static final Function<String, String> AS_FUNCTION = INSTANCE.asFunction();

  private static final CharMatcher UNSAFECHAR_MATCHER =
      CharMatcher.anyOf("'\"\\ \t\r\n\f\u000B").precomputed();

  @Override
  public String escape(String string) {
    if (string.isEmpty()) {
      // Empty string is a special case: needs to be quoted to ensure that it
      // gets treated as a separate argument.
      return "''";
    } else {
      return super.escape(string);
    }
  }

  @Override
  public char[] escape(char c) {
    if (!UNSAFECHAR_MATCHER.matches(c)) {
      return null;
    } else {
      char[] result = new char[2];
      result[0] = '\\';
      result[1] = c;
      return result;
    }
  }

  public static String escapeString(String unescaped) {
    return INSTANCE.escape(unescaped);
  }

  /**
   * Transforms the input {@code Iterable} of unescaped strings to an {@code Iterable} of escaped
   * ones. The escaping is done lazily.
   */
  public static Iterable<String> escapeAll(Iterable<? extends String> unescaped) {
    return Iterables.transform(unescaped, AS_FUNCTION);
  }
}
