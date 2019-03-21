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

package com.google.devtools.build.lib.util;

import com.google.common.base.CharMatcher;
import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.escape.CharEscaperBuilder;
import com.google.common.escape.Escaper;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import java.io.IOException;

/**
 * Utility class to escape strings for use with shell commands.
 *
 * <p>Escaped strings may safely be inserted into shell commands. Escaping is
 * only done if necessary. Strings containing only shell-neutral characters
 * will not be escaped.
 *
 * <p>This is a replacement for {@code ShellUtils.shellEscape(String)} and
 * {@code ShellUtils.prettyPrintArgv(java.util.List)} (see
 * {@link com.google.devtools.build.lib.shell.ShellUtils}). Its advantage is the use
 * of standard building blocks from the {@code com.google.common.base}
 * package, such as {@link Joiner} and {@link CharMatcher}, making this class
 * more efficient and reliable than {@code ShellUtils}.
 *
 * <p>The behavior is slightly different though: this implementation will
 * defensively escape non-ASCII letters and digits, whereas
 * {@code shellEscape} does not.
 */
@Immutable
public final class ShellEscaper extends Escaper {
  // Note: extending Escaper may seem desirable, but is in fact harmful.
  // The class would then need to implement escape(Appendable), returning an Appendable
  // that escapes everything it receives. In case of shell escaping, we most often join
  // string parts on spaces, using a Joiner. Spaces are escaped characters. Using the
  // Appendable returned by escape(Appendable) would escape these spaces too, which
  // is unwanted.

  public static final ShellEscaper INSTANCE = new ShellEscaper();

  private static final Function<String, String> AS_FUNCTION = INSTANCE.asFunction();

  private static final Joiner SPACE_JOINER = Joiner.on(' ');
  private static final Escaper STRONGQUOTE_ESCAPER =
      new CharEscaperBuilder().addEscape('\'', "'\\''").toEscaper();
  private static final CharMatcher SAFECHAR_MATCHER =
      CharMatcher.anyOf("@%-_+:,./")
          .or(CharMatcher.inRange('0', '9')) // We can't use CharMatcher.javaLetterOrDigit(),
          .or(CharMatcher.inRange('a', 'z')) // that would also accept non-ASCII digits and
          .or(CharMatcher.inRange('A', 'Z')) // letters.
          .precomputed();

  /**
   * Escapes a string by adding strong (single) quotes around it if necessary.
   *
   * <p>A string is not escaped iff it only contains safe characters.
   * The following characters are safe:
   * <ul>
   * <li>ASCII letters and digits: [a-zA-Z0-9]
   * <li>shell-neutral characters: at symbol (@), percent symbol (%),
   *     dash/minus sign (-), underscore (_), plus sign (+), colon (:),
   *     comma(,), period (.) and slash (/).
   * </ul>
   *
   * <p>A string is escaped iff it contains at least one non-safe character.
   * Escaped strings are created by replacing every occurrence of single
   * quotes with the string '\'' and enclosing the result in a pair of
   * single quotes.
   *
   * <p>Examples:
   * <ul>
   * <li>"{@code foo}" becomes "{@code foo}" (remains the same)
   * <li>"{@code +bar}" becomes "{@code +bar}" (remains the same)
   * <li>"" becomes "{@code''}" (empty string becomes a pair of strong quotes)
   * <li>"{@code $BAZ}" becomes "{@code '$BAZ'}"
   * <li>"{@code quote'd}" becomes "{@code 'quote'\''d'}"
   * </ul>
   */
  @Override
  public String escape(String unescaped) {
    final String s = unescaped.toString();
    if (s.isEmpty()) {
      // Empty string is a special case: needs to be quoted to ensure that it
      // gets treated as a separate argument.
      return "''";
    } else {
      return SAFECHAR_MATCHER.matchesAllOf(s)
          ? s
          : "'" + STRONGQUOTE_ESCAPER.escape(s) + "'";
    }
  }

  public static String escapeString(String unescaped) {
    return INSTANCE.escape(unescaped);
  }

  /**
   * Transforms the input {@code Iterable} of unescaped strings to an
   * {@code Iterable} of escaped ones. The escaping is done lazily.
   */
  public static Iterable<String> escapeAll(Iterable<? extends String> unescaped) {
    return Iterables.transform(unescaped, AS_FUNCTION);
  }

  /**
   * Escapes all strings in {@code argv} individually and joins them on
   * single spaces into {@code out}. The result is appended directly into
   * {@code out}, without adding a separator.
   *
   * <p>This method works as if by invoking
   * {@link #escapeJoinAll(Appendable, Iterable, Joiner)} with
   * {@code Joiner.on(' ')}.
   *
   * @param out what the result will be appended to
   * @param argv the strings to escape and join
   * @return the same reference as {@code out}, now containing the
   *     joined, escaped fragments
   * @throws IOException if an I/O error occurs while appending
   */
  public static Appendable escapeJoinAll(Appendable out, Iterable<? extends String> argv)
      throws IOException {
    return SPACE_JOINER.appendTo(out, escapeAll(argv));
  }

  /**
   * Escapes all strings in {@code argv} individually and joins them into
   * {@code out} using the specified {@link Joiner}. The result is appended
   * directly into {@code out}, without adding a separator.
   *
   * <p>The resulting strings are the same as if escaped one by one using
   * {@link #escapeString(String)}.
   *
   * <p>Example: if the joiner is {@code Joiner.on('|')}, then the input
   * {@code ["abc", "de'f"]} will be escaped as "{@code abc|'de'\''f'}".
   * If {@code out} initially contains "{@code 123}", then the returned
   * {@code Appendable} will contain "{@code 123abc|'de'\''f'}".
   *
   * @param out what the result will be appended to
   * @param argv the strings to escape and join
   * @param joiner the {@link Joiner} to use to join the escaped strings
   * @return the same reference as {@code out}, now containing the
   *     joined, escaped fragments
   * @throws IOException if an I/O error occurs while appending
   */
  public static Appendable escapeJoinAll(Appendable out, Iterable<? extends String> argv,
      Joiner joiner) throws IOException {
    return joiner.appendTo(out, escapeAll(argv));
  }

  /**
   * Escapes all strings in {@code argv} individually and joins them on
   * single spaces, then returns the resulting string.
   *
   * <p>This method works as if by invoking
   * {@link #escapeJoinAll(Iterable, Joiner)} with {@code Joiner.on(' ')}.
   *
   * <p>Example: {@code ["abc", "de'f"]} will be escaped and joined as
   * "abc 'de'\''f'".
   *
   * @param argv the strings to escape and join
   * @return the string of escaped and joined input elements
   */
  public static String escapeJoinAll(Iterable<? extends String> argv) {
    return SPACE_JOINER.join(escapeAll(argv));
  }

  /**
   * Escapes all strings in {@code argv} individually and joins them using
   * the specified {@link Joiner}, then returns the resulting string.
   *
   * <p>The resulting strings are the same as if escaped one by one using
   * {@link #escapeString(String)}.
   *
   * <p>Example: if the joiner is {@code Joiner.on('|')}, then the input
   * {@code ["abc", "de'f"]} will be escaped and joined as "abc|'de'\''f'".
   *
   * @param argv the strings to escape and join
   * @param joiner the {@link Joiner} to use to join the escaped strings
   * @return the string of escaped and joined input elements
   */
  public static String escapeJoinAll(Iterable<? extends String> argv, Joiner joiner) {
    return joiner.join(escapeAll(argv));
  }

  private ShellEscaper() {
    // Utility class - do not instantiate.
  }
}
