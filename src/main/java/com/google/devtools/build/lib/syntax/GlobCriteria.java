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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Functions;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableList.Builder;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.List;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/**
 * Either the arguments to a glob call (the include and exclude lists) or the
 * contents of a fixed list that was appended to a list of glob results.
 * (The latter need to be stored by {@link GlobList} in order to fully
 * reproduce the inputs that created the output list.)
 *
 * <p>For example, the expression
 * <code>glob(['*.java']) + ['x.properties']</code>
 * will result in two GlobCriteria: one has include = ['*.java'], glob = true
 * and the other, include = ['x.properties'], glob = false.
 */
public class GlobCriteria {

  /**
   * A list of names or patterns that are included by this glob. They should
   * consist of characters that are valid in labels in the BUILD language.
   */
  private final ImmutableList<String> include;

  /**
   * A list of names or patterns that are excluded by this glob. They should
   * consist of characters that are valid in labels in the BUILD language.
   */
  private final ImmutableList<String> exclude;

  /** True if the includes list was passed to glob(), false if not. */
  private final boolean glob;

  /**
   * Parses criteria from its {@link #toExpression} form.
   * Package-private for use by tests and GlobList.
   * @throws IllegalArgumentException if the expression cannot be parsed
   */
  public static GlobCriteria parse(String text) {
    if (text.startsWith("glob([") && text.endsWith("])")) {
      int excludeIndex = text.indexOf("], exclude=[");
      if (excludeIndex == -1) {
        String listText = text.substring(6, text.length() - 2);
        return new GlobCriteria(parseList(listText), ImmutableList.<String>of(), true);
      } else {
        String listText = text.substring(6, excludeIndex);
        String excludeText = text.substring(excludeIndex + 12, text.length() - 2);
        return new GlobCriteria(parseList(listText), parseList(excludeText), true);
      }
    } else if (text.startsWith("[") && text.endsWith("]")) {
      String listText = text.substring(1, text.length() - 1);
      return new GlobCriteria(parseList(listText), ImmutableList.<String>of(), false);
    } else {
      throw new IllegalArgumentException(
          "unrecognized format (not from toExpression?): " + text);
    }
  }

  /**
   * Constructs a copy of a given glob critera object, with additional exclude patterns added.
   *
   * @param base a glob criteria object to copy. Must be an actual glob
   * @param excludes a list of pattern strings indicating new excludes to provide
   * @return a new glob criteria object which contains the same parameters as {@code base}, with
   *   the additional patterns in {@code excludes} added.
   * @throws IllegalArgumentException if {@code base} is not a glob
   */
  public static GlobCriteria createWithAdditionalExcludes(GlobCriteria base,
      List<String> excludes) {
    Preconditions.checkArgument(base.isGlob());
    return fromGlobCall(
        base.include,
        Stream.concat(base.exclude.stream(), excludes.stream()).collect(toImmutableList()));
  }

  /**
   * Constructs a copy of a fixed list, converted to Strings.
   */
  public static GlobCriteria fromList(Iterable<?> list) {
    Iterable<String> strings = Iterables.transform(list, Functions.toStringFunction());
    return new GlobCriteria(ImmutableList.copyOf(strings), ImmutableList.<String>of(), false);
  }

  /**
   * Constructs a glob call with include and exclude list.
   *
   * @param include list of included patterns
   * @param exclude list of excluded patterns
   */
  public static GlobCriteria fromGlobCall(
      ImmutableList<String> include, ImmutableList<String> exclude) {
    return new GlobCriteria(include, exclude, true);
  }

  /**
   * Constructs a glob call with include and exclude list.
   */
  private GlobCriteria(ImmutableList<String> include, ImmutableList<String> exclude, boolean glob) {
    this.include = include;
    this.exclude = exclude;
    this.glob = glob;
  }

  /**
   * Returns the patterns that were included in this {@code glob()} call.
   */
  public ImmutableList<String> getIncludePatterns() {
    return include;
  }

  /**
   * Returns the patterns that were excluded in this {@code glob()} call.
   */
  public ImmutableList<String> getExcludePatterns() {
    return exclude;
  }

  /**
   * Returns true if the include list was passed to {@code glob()}, false
   * if it was a fixed list. If this returns false, the exclude list will
   * always be empty.
   */
  public boolean isGlob() {
    return glob;
  }

  /**
   * Returns a String that represents this glob as a BUILD expression.
   * For example, <code>glob(['abc', 'def'], exclude=['uvw', 'xyz'])</code>
   * or <code>['foo', 'bar', 'baz']</code>.
   */
  public String toExpression() {
    StringBuilder sb = new StringBuilder();
    if (glob) {
      sb.append("glob(");
    }
    sb.append('[');
    appendList(sb, include);
    if (!exclude.isEmpty()) {
      sb.append("], exclude=[");
      appendList(sb, exclude);
    }
    sb.append(']');
    if (glob) {
      sb.append(')');
    }
    return sb.toString();
  }

  @Override
  public String toString() {
    return toExpression();
  }

  /**
   * Takes a list of Strings, quotes them in single quotes, and appends them to
   * a StringBuilder separated by a comma and space. This can be parsed back
   * out by {@link #parseList}.
   */
  private static void appendList(StringBuilder sb, List<String> list) {
    boolean first = true;
    for (String content : list) {
      if (!first) {
        sb.append(", ");
      }
      sb.append('\'').append(content).append('\'');
      first = false;
    }
  }

  /**
   * Takes a String in the format created by {@link #appendList} and returns
   * the original Strings. A null String (which may be returned when Pattern
   * does not find a match) or the String "" (which will be captured in "[]")
   * will result in an empty list.
   */
  private static ImmutableList<String> parseList(@Nullable String text) {
    if (text == null) {
      return ImmutableList.of();
    }
    Iterable<String> split = Splitter.on(", ").split(text);
    Builder<String> listBuilder = ImmutableList.builder();
    for (String element : split) {
      if (!element.isEmpty()) {
        if ((element.length() < 2) || !element.startsWith("'") || !element.endsWith("'")) {
          throw new IllegalArgumentException("expected a filename or pattern in quotes: " + text);
        }
        listBuilder.add(element.substring(1, element.length() - 1));
      }
    }
    return listBuilder.build();
  }
}
