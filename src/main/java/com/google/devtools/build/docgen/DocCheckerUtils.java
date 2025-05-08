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

package com.google.devtools.build.docgen;

import com.google.common.collect.ImmutableSet;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * A utility class to check the generated documentations.
 */
public class DocCheckerUtils {

  // TODO(bazel-team): remove elements from this list and clean up the tested documentations.
  private static final ImmutableSet<String> UNCHECKED_HTML_TAGS = ImmutableSet.<String>of(
      "br", "li", "ul", "p");

  private static final Pattern TAG_PATTERN = Pattern.compile(
        "<([/]?[a-z0-9_]+)"
      + "([^>]*)"
      + ">",
      Pattern.CASE_INSENSITIVE);

  private static final Pattern COMMENT_OR_BACKTICK_PATTERN =
      Pattern.compile("<!--.*?-->|`.*`", Pattern.CASE_INSENSITIVE);

  /**
   * Returns the first unmatched html tag of srcs or null if no such tag exists.
   * Note that this check is not performed on br, ul, li and p tags. The method also
   * prints some help in case an unmatched tag is found. The check is performed
   * inside comments too.
   */
  public static String getFirstUnclosedTagAndPrintHelp(String src) {
    return getFirstUnclosedTag(src, true);
  }

  static String getFirstUnclosedTag(String src) {
    return getFirstUnclosedTag(src, false);
  }

  // TODO(bazel-team): run this on the Starlark docs too.
  @Nullable
  private static String getFirstUnclosedTag(String src, boolean printHelp) {
    Matcher commentMatcher = COMMENT_OR_BACKTICK_PATTERN.matcher(src);
    src = commentMatcher.replaceAll("");
    Matcher tagMatcher = TAG_PATTERN.matcher(src);
    Deque<String> tagStack = new ArrayDeque<>();
    while (tagMatcher.find()) {
      String tag = tagMatcher.group(1);
      String rest = tagMatcher.group(2);
      String strippedTag = tag.substring(1);

      // Ignoring self closing tags.
      if (!rest.endsWith("/")
          // Ignoring unchecked tags.
          && !UNCHECKED_HTML_TAGS.contains(tag) && !UNCHECKED_HTML_TAGS.contains(strippedTag)) {
        if (tag.startsWith("/")) {
          // Closing tag. Removing '/' from the beginning.
          tag = strippedTag;
          String lastTag = tagStack.removeLast();
          if (!lastTag.equals(tag)) {
            if (printHelp) {
              System.err.println(
                    "Unclosed tag: " + lastTag + "\n"
                  + "Trying to close with: " + tag + "\n"
                  + "Stack of open tags: " + tagStack + "\n"
                  + "Last 200 characters:\n"
                  + src.substring(Math.max(tagMatcher.start() - 200, 0), tagMatcher.start()));
            }
            return lastTag;
          }
        } else {
          // Starting tag.
          tagStack.addLast(tag);
        }
      }
    }
    return null;
  }
}
