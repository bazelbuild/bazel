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

import java.io.PrintStream;
import java.util.ArrayDeque;
import java.util.Deque;

/**
 * Utility function for writing HTML data to a {@link PrintStream}.
 */
public abstract class HtmlPrinter extends TextPrinter {

  private Deque<String> currentlyOpenTags;

  protected HtmlPrinter(PrintStream out) {
    super(out);
    currentlyOpenTags = new ArrayDeque<>();
  }

  /**
   * Print an open tag with attributes and possibly content and increase indentation level.
   *
   * <p>All array elements are taken in pairs for attributes and their values. If odd, the last
   *  element is taken as the content of the element. It is printed directly after the opening tag.
   * @param attributesAndContent must have the form: attribute1, value1, attribute2, value2, ...,
   *    content
   */
  protected void open(String tag, Object... attributesAndContent) {
    printf("<%s", tag);
    for (int index = 0; index < attributesAndContent.length - 1; index += 2) {
      printf(" %s=\"%s\"", attributesAndContent[index], attributesAndContent[index + 1]);
    }
    print(">");
    if (attributesAndContent.length % 2 == 1) {
      print(attributesAndContent[attributesAndContent.length - 1]);
    }
    down();
    currentlyOpenTags.addFirst(tag);
  }

  /**
   * Print a newline, an open tag with attributes and possibly content and increase indentation
   * level
   * @see #open(String, Object...)
   */
  protected void lnOpen(String tag, Object... attributes) {
    lnIndent();
    open(tag, attributes);
  }

  /**
   * Decrease indentation level and close the most recently opened tag
   */
  protected void close() {
    up();
    printf("</%s>", currentlyOpenTags.pop());
  }

  /**
   * Decrease indentation level, print newline, indentation and close the most recently opened tag
   */
  protected void lnClose() {
    up();
    lnPrintf("</%s>", currentlyOpenTags.pop());
  }

  protected void lnElement(String tag, Object content) {
    lnIndent();
    element(tag, content);
  }

  /**
   * Print a newline, indent and a single element with attributes and possibly content.
   *
   * @see #lnOpen(String, Object...)
   */
  protected void lnElement(String tag, Object... attributesAndContent) {
    lnOpen(tag, attributesAndContent);
    close();
  }

  /**
   * Print a single element with attributes and possibly content.
   *
   * @see #open(String, Object...)
   */
  protected void element(String tag, Object... attributesAndContent) {
    open(tag, attributesAndContent);
    close();
  }
}

