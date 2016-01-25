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
package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrintableValue;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.Formattable;
import java.util.Formatter;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.Map;
import java.util.MissingFormatWidthException;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * (Pretty) Printing of Skylark values
 */
public final class Printer {

  private static final char SKYLARK_QUOTATION_MARK = '"';

  /*
   * Suggested maximum number of list elements that should be printed via printList().
   * By default, this setting is not considered and no limitation takes place.
   */
  static final int SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT = 4;

  /*
   * Suggested limit for printList() to shorten the values of list elements when their combined
   * string length reaches this value.
   * By default, this setting is not considered and no limitation takes place.
   */
  static final int SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH = 32;

  private Printer() {
  }

  /**
   * Get an informal representation of object x.
   * Currently only differs from repr in the behavior for strings and labels at top-level,
   * that are returned as is rather than quoted.
   * @param quotationMark The quotation mark to be used (' or ")
   * @return the representation.
   */
  public static String str(Object x, char quotationMark) {
    return print(new StringBuilder(), x, quotationMark).toString();
  }

  public static String str(Object x) {
    return str(x, SKYLARK_QUOTATION_MARK);
  }

  /**
   * Get an official representation of object x.
   * For regular data structures, the value should be parsable back into an equal data structure.
   * @param quotationMark The quotation mark to be used (' or ")
   * @return the representation.
   */
  public static String repr(Object x, char quotationMark) {
    return write(new StringBuilder(), x, quotationMark).toString();
  }

  public static String repr(Object x) {
    return repr(x, SKYLARK_QUOTATION_MARK);
  }

  // In absence of a Python naming tradition, the write() vs print() function names
  // follow the Lisp tradition: print() displays the informal representation (as in Python str)
  // whereas write() displays a readable representation (as in Python repr).
  /**
   * Print an informal representation of object x.
   * Currently only differs from repr in the behavior for strings and labels at top-level,
   * that are returned as is rather than quoted.
   * @param buffer the Appendable to which to print the representation
   * @param o the object
   * @param quotationMark The quotation mark to be used (' or ")
   * @return the buffer, in fluent style
   */
  private static Appendable print(Appendable buffer, Object o, char quotationMark) {
    if (o instanceof SkylarkPrintableValue) {
      ((SkylarkPrintableValue) o).print(buffer, quotationMark);
      return buffer;
    }

    if (o instanceof String) {
      return append(buffer, (String) o);
    }
    return write(buffer, o, quotationMark);
  }

  private static Appendable print(Appendable buffer, Object o) {
    return print(buffer, o, SKYLARK_QUOTATION_MARK);
  }

  /**
   * Print an official representation of object x.
   * For regular data structures, the value should be parsable back into an equal data structure.
   * @param buffer the Appendable to write to.
   * @param o the string a representation of which to write.
   * @param quotationMark The quotation mark to be used (' or ")
   * @return the Appendable, in fluent style.
   */
  public static Appendable write(Appendable buffer, Object o, char quotationMark) {
    if (o == null) {
      throw new NullPointerException(); // Java null is not a build language value.

    } else if (o instanceof SkylarkValue) {
      ((SkylarkValue) o).write(buffer, quotationMark);

    } else if (o instanceof String) {
      writeString(buffer, (String) o, quotationMark);

    } else if (o instanceof Integer || o instanceof Double) {
      append(buffer, o.toString());

    } else if (o == Boolean.TRUE) {
      append(buffer, "True");

    } else if (o == Boolean.FALSE) {
      append(buffer, "False");

    } else if (o instanceof List<?>) {
      List<?> seq = (List<?>) o;
      printList(buffer, seq, false, quotationMark);

    } else if (o instanceof Map<?, ?>) {
      Map<?, ?> dict = (Map<?, ?>) o;
      printList(buffer, getSortedEntrySet(dict), "{", ", ", "}", null, quotationMark);

    } else if (o instanceof Map.Entry<?, ?>) {
      Map.Entry<?, ?> entry = (Map.Entry<?, ?>) o;
      write(buffer, entry.getKey(), quotationMark);
      append(buffer, ": ");
      write(buffer, entry.getValue(), quotationMark);

    } else if (o instanceof PathFragment) {
      append(buffer, ((PathFragment) o).getPathString());

    } else if (o instanceof Class<?>) {
      append(buffer, EvalUtils.getDataTypeNameFromClass((Class<?>) o));

    } else {
      append(buffer, o.toString());
    }

    return buffer;
  }

  /**
   * Returns the sorted entry set of the given map
   */
  private static <K, V> Set<Map.Entry<K, V>> getSortedEntrySet(Map<K, V> dict) {
    if (!(dict instanceof SortedMap<?, ?>)) {
      Map<K, V> tmp = new TreeMap<>(EvalUtils.SKYLARK_COMPARATOR);
      tmp.putAll(dict);
      dict = tmp;
    }

    return dict.entrySet();
  }

  public static Appendable write(Appendable buffer, Object o) {
    return write(buffer, o, SKYLARK_QUOTATION_MARK);
  }

  // Throughout this file, we transform IOException into AssertionError.
  // During normal operations, we only use in-memory Appendable-s that
  // cannot cause an IOException.
  public static Appendable append(Appendable buffer, char c) {
    try {
      return buffer.append(c);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  public static Appendable append(Appendable buffer, CharSequence s) {
    try {
      return buffer.append(s);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  private static Appendable append(Appendable buffer, CharSequence s, int start, int end) {
    try {
      return buffer.append(s, start, end);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  private static Appendable backslashChar(Appendable buffer, char c) {
    return append(append(buffer, '\\'), c);
  }

  private static Appendable escapeCharacter(Appendable buffer, char c, char quote) {
    if (c == quote) {
      return backslashChar(buffer, c);
    }
    switch (c) {
      case '\\':
        return backslashChar(buffer, '\\');
      case '\r':
        return backslashChar(buffer, 'r');
      case '\n':
        return backslashChar(buffer, 'n');
      case '\t':
        return backslashChar(buffer, 't');
      default:
        if (c < 32) {
          return append(buffer, String.format("\\x%02x", (int) c));
        }
        return append(buffer, c); // no need to support UTF-8
    } // endswitch
  }

  /**
   * Write a properly escaped Skylark representation of a string to a buffer.
   *
   * @param buffer the Appendable to write to.
   * @param s the string a representation of which to write.
   * @param quote the quote character to use, '"' or '\''.
   * @return the Appendable, in fluent style.
   */
  private static Appendable writeString(Appendable buffer, String s, char quote) {
    append(buffer, quote);
    int len = s.length();
    for (int i = 0; i < len; i++) {
      char c = s.charAt(i);
      escapeCharacter(buffer, c, quote);
    }
    return append(buffer, quote);
  }

  /**
   * Print a list of object representations
   * @param buffer an appendable buffer onto which to write the list.
   * @param list the list of objects to write (each as with repr)
   * @param before a string to print before the list
   * @param separator a separator to print between each object
   * @param after a string to print after the list
   * @param singletonTerminator null or a string to print after the list if it is a singleton
   * The singleton case is notably relied upon in python syntax to distinguish
   * a tuple of size one such as ("foo",) from a merely parenthesized object such as ("foo").
   * @param quotationMark The quotation mark to be used (' or ")
   * @return the Appendable, in fluent style.
   */
  public static Appendable printList(
      Appendable buffer,
      Iterable<?> list,
      String before,
      String separator,
      String after,
      String singletonTerminator,
      char quotationMark) {
    return printList(
        buffer, list, before, separator, after, singletonTerminator, quotationMark, -1, -1);
  }

  /**
   * Print a list of object representations.
   *
   * <p>The length of the output will be limited when both {@code maxItemsToPrint} and {@code
   * criticalItemsStringLength} have values greater than zero.
   *
   * @param buffer an appendable buffer onto which to write the list.
   * @param list the list of objects to write (each as with repr)
   * @param before a string to print before the list
   * @param separator a separator to print between each object
   * @param after a string to print after the list
   * @param singletonTerminator null or a string to print after the list if it is a singleton
   * The singleton case is notably relied upon in python syntax to distinguish
   *    a tuple of size one such as ("foo",) from a merely parenthesized object such as ("foo").
   * @param quotationMark The quotation mark to be used (' or ")
   * @param maxItemsToPrint the maximum number of elements to be printed.
   * @param criticalItemsStringLength a soft limit for the total string length of all arguments.
   *    'Soft' means that this limit may be exceeded because of formatting.
   * @return the Appendable, in fluent style.
   */
  public static Appendable printList(Appendable buffer, Iterable<?> list, String before,
      String separator, String after, String singletonTerminator, char quotationMark,
      int maxItemsToPrint, int criticalItemsStringLength) {
    append(buffer, before);
    int len = 0;
    // Limits the total length of the string representation of the elements, if specified.
    if (maxItemsToPrint > 0 && criticalItemsStringLength > 0) {
      len = appendListElements(LengthLimitedAppendable.create(buffer, criticalItemsStringLength),
          list, separator, quotationMark, maxItemsToPrint);
    } else {
      len = appendListElements(buffer, list, separator, quotationMark);
    }
    if (singletonTerminator != null && len == 1) {
      append(buffer, singletonTerminator);
    }
    return append(buffer, after);
  }

  public static Appendable printList(Appendable buffer, Iterable<?> list, String before,
      String separator, String after, String singletonTerminator, int maxItemsToPrint,
      int criticalItemsStringLength) {
    return printList(buffer, list, before, separator, after, singletonTerminator,
        SKYLARK_QUOTATION_MARK, maxItemsToPrint, criticalItemsStringLength);
  }

  /**
   * Appends the given elements to the specified {@link Appendable} and returns the number of
   * elements.
   */
  private static int appendListElements(
      Appendable appendable, Iterable<?> list, String separator, char quotationMark) {
    boolean printSeparator = false; // don't print the separator before the first element
    int len = 0;
    for (Object o : list) {
      if (printSeparator) {
        append(appendable, separator);
      }
      write(appendable, o, quotationMark);
      printSeparator = true;
      len++;
    }
    return len;
  }

  /**
   * Tries to append the given elements to the specified {@link Appendable} until specific limits
   * are reached.
   * @return the number of appended elements.
   */
  private static int appendListElements(LengthLimitedAppendable appendable, Iterable<?> list,
      String separator, char quotationMark, int maxItemsToPrint) {
    boolean printSeparator = false; // don't print the separator before the first element
    boolean skipArgs = false;
    int items = Iterables.size(list);
    int len = 0;
    // We don't want to print "1 more arguments", hence we don't skip arguments if there is only one
    // above the limit.
    int itemsToPrint = (items - maxItemsToPrint == 1) ? items : maxItemsToPrint;
    appendable.enforceLimit();
    for (Object o : list) {
      // We don't want to print "1 more arguments", even if we hit the string limit.
      if (len == itemsToPrint || (appendable.hasHitLimit() && len < items - 1)) {
        skipArgs = true;
        break;
      }
      if (printSeparator) {
        append(appendable, separator);
      }
      write(appendable, o, quotationMark);
      printSeparator = true;
      len++;
    }
    appendable.ignoreLimit();
    if (skipArgs) {
      append(appendable, separator);
      append(appendable, String.format("<%d more arguments>", items - len));
    }
    return len;
  }

  public static Appendable printList(Appendable buffer, Iterable<?> list, String before,
      String separator, String after, String singletonTerminator) {
    return printList(
        buffer, list, before, separator, after, singletonTerminator, SKYLARK_QUOTATION_MARK);
  }

  /**
   * Print a Skylark list or tuple of object representations
   * @param buffer an appendable buffer onto which to write the list.
   * @param list the contents of the list or tuple
   * @param isTuple is it a tuple or a list?
   * @param quotationMark The quotation mark to be used (' or ")
   * @param maxItemsToPrint the maximum number of elements to be printed.
   * @param criticalItemsStringLength a soft limit for the total string length of all arguments.
   * 'Soft' means that this limit may be exceeded because of formatting.
   * @return the Appendable, in fluent style.
   */
  public static Appendable printList(Appendable buffer, Iterable<?> list, boolean isTuple,
      char quotationMark, int maxItemsToPrint, int criticalItemsStringLength) {
    if (isTuple) {
      return printList(buffer, list, "(", ", ", ")", ",", quotationMark, maxItemsToPrint,
          criticalItemsStringLength);
    } else {
      return printList(buffer, list, "[", ", ", "]", null, quotationMark, maxItemsToPrint,
          criticalItemsStringLength);
    }
  }

  public static Appendable printList(
      Appendable buffer, Iterable<?> list, boolean isTuple, char quotationMark) {
    return printList(buffer, list, isTuple, quotationMark, -1, -1);
  }

  /**
   * Print a list of object representations
   * @param list the list of objects to write (each as with repr)
   * @param before a string to print before the list
   * @param separator a separator to print between each object
   * @param after a string to print after the list
   * @param singletonTerminator null or a string to print after the list if it is a singleton
   * The singleton case is notably relied upon in python syntax to distinguish
   * a tuple of size one such as ("foo",) from a merely parenthesized object such as ("foo").
   * @param quotationMark The quotation mark to be used (' or ")
   * @return a String, the representation.
   */
  public static String listString(Iterable<?> list, String before, String separator, String after,
      String singletonTerminator, char quotationMark) {
    return printList(new StringBuilder(), list, before, separator, after, singletonTerminator,
               quotationMark).toString();
  }

  public static String listString(
      Iterable<?> list, String before, String separator, String after, String singletonTerminator) {
    return listString(list, before, separator, after, singletonTerminator, SKYLARK_QUOTATION_MARK);
  }

  /**
   * Perform Python-style string formatting, lazily.
   *
   * @param pattern a format string.
   * @param arguments positional arguments.
   * @return the formatted string.
   */
  public static Formattable formattable(final String pattern, Object... arguments)
      throws IllegalFormatException {
    final ImmutableList<Object> args = ImmutableList.copyOf(arguments);
    return new Formattable() {
        @Override
        public String toString() {
          return formatToString(pattern, args);
        }

        @Override
        public void formatTo(Formatter formatter, int flags, int width, int precision) {
          Printer.formatTo(formatter.out(), pattern, args);
        }
      };
  }

  /**
   * Perform Python-style string formatting.
   *
   * @param pattern a format string.
   * @param arguments a tuple containing positional arguments.
   * @return the formatted string.
   */
  public static String format(String pattern, Object... arguments)
      throws IllegalFormatException {
    return formatToString(pattern, ImmutableList.copyOf(arguments));
  }

  /**
   * Perform Python-style string formatting.
   *
   * @param pattern a format string.
   * @param arguments a tuple containing positional arguments.
   * @return the formatted string.
   */
  public static String formatToString(String pattern, List<?> arguments)
      throws IllegalFormatException {
    return formatTo(new StringBuilder(), pattern, arguments).toString();
  }

  /**
   * Perform Python-style string formatting, as per pattern % tuple
   * Limitations: only %d %s %r %% are supported.
   *
   * @param buffer an Appendable to output to.
   * @param pattern a format string.
   * @param arguments a list containing positional arguments.
   * @return the buffer, in fluent style.
   */
  // TODO(bazel-team): support formatting arguments, and more complex Python patterns.
  public static Appendable formatTo(Appendable buffer, String pattern, List<?> arguments)
      throws IllegalFormatException {
    // N.B. MissingFormatWidthException is the only kind of IllegalFormatException
    // whose constructor can take and display arbitrary error message, hence its use below.

    int length = pattern.length();
    int argLength = arguments.size();
    int i = 0; // index of next character in pattern
    int a = 0; // index of next argument in arguments

    while (i < length) {
      int p = pattern.indexOf('%', i);
      if (p == -1) {
        append(buffer, pattern, i, length);
        break;
      }
      if (p > i) {
        append(buffer, pattern, i, p);
      }
      if (p == length - 1) {
        throw new MissingFormatWidthException(
            "incomplete format pattern ends with %: " + repr(pattern));
      }
      char directive = pattern.charAt(p + 1);
      i = p + 2;
      switch (directive) {
        case '%':
          append(buffer, '%');
          continue;
        case 'd':
        case 'r':
        case 's':
          if (a >= argLength) {
            throw new MissingFormatWidthException("not enough arguments for format pattern "
                + repr(pattern) + ": "
                + repr(Tuple.copyOf(arguments)));
          }
          Object argument = arguments.get(a++);
          switch (directive) {
            case 'd':
              if (argument instanceof Integer) {
                append(buffer, argument.toString());
                continue;
              } else {
                throw new MissingFormatWidthException(
                    "invalid argument " + repr(argument) + " for format pattern %d");
              }
            case 'r':
              write(buffer, argument);
              continue;
            case 's':
              print(buffer, argument);
              continue;
          }
        default:
          throw new MissingFormatWidthException(
              "unsupported format character " + repr(String.valueOf(directive))
              + " at index " + (p + 1) + " in " + repr(pattern));
      }
    }
    if (a < argLength) {
      throw new MissingFormatWidthException(
          "not all arguments converted during string formatting");
    }
    return buffer;
  }

  /**
   * Helper class for {@code Appendable}s that want to limit the length of their input.
   *
   * <p>Instances of this class act as a proxy for one {@code Appendable} object and decide whether
   * the given input (or parts of it) can be written to the underlying {@code Appendable}, depending
   * on whether the specified maximum length has been met or not.
   */
  private static final class LengthLimitedAppendable implements Appendable {

    private static final ImmutableSet<Character> SPECIAL_CHARS =
        ImmutableSet.of(',', ' ', '"', '\'', ':', '(', ')', '[', ']', '{', '}');

    private static final Pattern ARGS_PATTERN = Pattern.compile("<\\d+ more arguments>");

    private final Appendable original;
    private int limit;
    private boolean ignoreLimit;
    private boolean previouslyShortened;
    
    private LengthLimitedAppendable(Appendable original, int limit) {
      this.original = original;
      this.limit = limit;
    }

    public static LengthLimitedAppendable create(Appendable original, int limit) {
      // We don't want to overwrite the limit if original is already an instance of this class.
      return (original instanceof LengthLimitedAppendable)
          ? (LengthLimitedAppendable) original : new LengthLimitedAppendable(original, limit);
    }

    @Override
    public Appendable append(CharSequence csq) throws IOException {
      if (ignoreLimit || hasOnlySpecialChars(csq)) {
        // Don't update limit.
        original.append(csq);
        previouslyShortened = false;
      } else {
        int length = csq.length();
        if (length <= limit) {
          limit -= length;
          original.append(csq);
        } else {
          original.append(csq, 0, limit);
          // We don't want to append multiple ellipses.
          if (!previouslyShortened) {
            original.append("...");
          }
          appendTrailingSpecialChars(csq, limit);
          previouslyShortened = true;
          limit = 0;
        }
      }
      return this;
    }

    /**
     * Appends any trailing "special characters" (e.g. brackets, quotation marks) in the given
     * sequence to the output buffer, regardless of the limit.
     *
     * <p>For example, let's look at foo(['too long']). Without this method, the shortened result
     * would be foo(['too...) instead of the prettier foo(['too...']).
     *
     * <p>If the input string was already shortened and contains "<x more arguments>", this part
     * will also be appended.
     */
    private void appendTrailingSpecialChars(CharSequence csq, int limit) throws IOException {
      int length = csq.length();
      Matcher matcher = ARGS_PATTERN.matcher(csq);
      // We assume that everything following the "x more arguments" part has to be copied, too.
      int start = matcher.find() ? matcher.start() : length;
      // Find the left-most non-arg char that has to be copied.
      for (int i = start - 1; i > limit; --i) {
        if (isSpecialChar(csq.charAt(i))) {
          start = i;
        } else {
          break;
        }
      }
      if (start < length) {
        original.append(csq, start, csq.length());
      }
    }

    /**
     * Returns whether the given sequence denotes characters that are not part of the value of an
     * argument.
     *
     * <p>Examples are brackets, braces and quotation marks.
     */
    private boolean hasOnlySpecialChars(CharSequence csq) {
      for (int i = 0; i < csq.length(); ++i) {
        if (!isSpecialChar(csq.charAt(i))) {
          return false;
        }
      }
      return true;
    }

    private boolean isSpecialChar(char c)    {
      return SPECIAL_CHARS.contains(c);
    }

    @Override
    public Appendable append(CharSequence csq, int start, int end) throws IOException {
      return append(csq.subSequence(start, end));
    }

    @Override
    public Appendable append(char c) throws IOException {
      return append(String.valueOf(c));
    }
    
    public boolean hasHitLimit()  {
      return limit <= 0;
    }

    public void enforceLimit()  {
      ignoreLimit = false;
    }
    
    public void ignoreLimit() {
      ignoreLimit = true;
    }

    @Override
    public String toString() {
      return original.toString();
    }
  }
}
