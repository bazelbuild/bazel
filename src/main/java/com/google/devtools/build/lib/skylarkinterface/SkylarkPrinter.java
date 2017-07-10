// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkinterface;

import java.util.List;
import javax.annotation.Nullable;

/**
 * An interface for Printer.BasePrinter
 *
 * <p>Allows passing Printer.BasePrinter instances to classes that can't import Printer directly
 * because of circular dependencies.
 */
public interface SkylarkPrinter {
  /** Append a char to the printer's buffer */
  SkylarkPrinter append(char c);

  /** Append a char sequence to the printer's buffer */
  SkylarkPrinter append(CharSequence s);

  /**
   * Prints a list to the printer's buffer. All list items are rendered with {@code repr}.
   *
   * @param list the list
   * @param isTuple if true, uses parentheses, otherwise, uses square brackets. Also one-element
   *     tuples are rendered with a comma after the element.
   * @return SkylarkPrinter
   */
  SkylarkPrinter printList(Iterable<?> list, boolean isTuple);

  /**
   * Prints a list to the printer's buffer. All list items are rendered with {@code repr}.
   *
   * @param list the list of objects to repr (each as with repr)
   * @param before a string to print before the list items, e.g. an opening bracket
   * @param separator a separator to print between items
   * @param after a string to print after the list items, e.g. a closing bracket
   * @param singletonTerminator null or a string to print after the list if it is a singleton.
   *     The singleton case is notably relied upon in python syntax to distinguish a tuple of size
   *     one such as ("foo",) from a merely parenthesized object such as ("foo")
   * @return SkylarkPrinter
   */
  SkylarkPrinter printList(
      Iterable<?> list, String before, String separator, String after,
      @Nullable String singletonTerminator);

  /**
   * Renders an object with {@code repr} and append to the printer's buffer.
   */
  SkylarkPrinter repr(Object o);

  /**
   * Performs Python-style string formatting, as per {@code pattern % tuple}.
   * Limitations: only {@code %d %s %r %%} are supported.
   *
   * @param pattern a format string
   * @param arguments an array containing positional arguments
   * @return SkylarkPrinter
   */
  SkylarkPrinter format(String pattern, Object... arguments);

  /**
   * Performs Python-style string formatting, as per {@code pattern % tuple}.
   * Limitations: only {@code %d %s %r %%} are supported.
   *
   * @param pattern a format string
   * @param arguments a list containing positional arguments
   * @return SkylarkPrinter
   */
  SkylarkPrinter formatWithList(String pattern, List<?> arguments);
}
