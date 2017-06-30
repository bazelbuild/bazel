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
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrintableValue;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Formattable;
import java.util.Formatter;
import java.util.List;
import java.util.Map;
import java.util.MissingFormatWidthException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** (Pretty) Printing of Skylark values */
public class Printer {

  public static final char SKYLARK_QUOTATION_MARK = '"';

  /*
   * Suggested maximum number of list elements that should be printed via printAbbreviatedList().
   * By default, this setting is not considered and no limitation takes place.
   */
  public static final int SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT = 4;

  /*
   * Suggested limit for printAbbreviatedList() to shorten the values of list elements when
   * their combined string length reaches this value.
   * By default, this setting is not considered and no limitation takes place.
   */
  public static final int SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH = 32;

  /**
   * Creates an instance of BasePrinter that wraps an existing buffer.
   * @param buffer an Appendable
   * @return new BasePrinter
   */
  public static BasePrinter getPrinter(Appendable buffer) {
    return new BasePrinter(buffer);
  }

  /**
   * Creates an instance of BasePrinter with an empty buffer.
   * @return new BasePrinter
   */
  public static BasePrinter getPrinter() {
    return getPrinter(new StringBuilder());
  }

  private Printer() {}

  // These static methods proxy to the similar methods of BasePrinter

  /**
   * Format an object with Skylark's {@code str}.
   */
  public static String str(Object x) {
    return getPrinter().str(x).toString();
  }

  /**
   * Format an object with Skylark's {@code repr}.
   */
  public static String repr(Object x) {
    return getPrinter().repr(x).toString();
  }

  /**
   * Print a list of object representations.
   *
   * <p>The length of the output will be limited when both {@code maxItemsToPrint} and
   * {@code criticalItemsStringLength} have values greater than zero.
   *
   * @param list the list of objects to repr (each as with repr)
   * @param before a string to print before the list
   * @param separator a separator to print between each object
   * @param after a string to print after the list
   * @param singletonTerminator null or a string to print after the list if it is a singleton The
   *     singleton case is notably relied upon in python syntax to distinguish a tuple of size one
   *     such as ("foo",) from a merely parenthesized object such as ("foo").
   * @param maxItemsToPrint the maximum number of elements to be printed.
   * @param criticalItemsStringLength a soft limit for the total string length of all arguments.
   *     'Soft' means that this limit may be exceeded because of formatting.
   * @return string representation.
   */
  public static String printAbbreviatedList(
      Iterable<?> list,
      String before,
      String separator,
      String after,
      @Nullable String singletonTerminator,
      int maxItemsToPrint,
      int criticalItemsStringLength) {
    return getPrinter()
        .printAbbreviatedList(
            list,
            before,
            separator,
            after,
            singletonTerminator,
            maxItemsToPrint,
            criticalItemsStringLength)
        .toString();
  }

  /**
   * Print a list of object representations.
   *
   * @param list the list of objects to repr (each as with repr)
   * @param before a string to print before the list
   * @param separator a separator to print between each object
   * @param after a string to print after the list
   * @param singletonTerminator null or a string to print after the list if it is a singleton The
   *     singleton case is notably relied upon in python syntax to distinguish a tuple of size one
   *     such as ("foo",) from a merely parenthesized object such as ("foo").
   * @return string representation.
   */
    public static String printAbbreviatedList(
      Iterable<?> list,
      String before,
      String separator,
      String after,
      @Nullable String singletonTerminator) {
    return printAbbreviatedList(list, before, separator, after, singletonTerminator,
        SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT, SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH);
  }

  /**
   * Print a list of object representations.
   *
   * <p>The length of the output will be limited when both {@code maxItemsToPrint} and
   * {@code criticalItemsStringLength} have values greater than zero.
   *
   * @param list the list of objects to repr (each as with repr)
   * @param isTuple if true the list will be formatted with parentheses and with a trailing comma
   *     in case of one-element tuples.
   * @param maxItemsToPrint the maximum number of elements to be printed.
   * @param criticalItemsStringLength a soft limit for the total string length of all arguments.
   *     'Soft' means that this limit may be exceeded because of formatting.
   * @return string representation.
   */
  public static String printAbbreviatedList(
      Iterable<?> list,
      boolean isTuple,
      int maxItemsToPrint,
      int criticalItemsStringLength) {
    return getPrinter()
        .printAbbreviatedList(list, isTuple, maxItemsToPrint, criticalItemsStringLength)
        .toString();
  }

  /**
   * Perform Python-style string formatting, as per pattern % tuple Limitations: only %d %s %r %%
   * are supported.
   *
   * @param pattern a format string.
   * @param arguments an array containing positional arguments.
   * @return the formatted string.
   */
  public static String format(String pattern, Object... arguments) {
    return getPrinter().format(pattern, arguments).toString();
  }

  /**
   * Perform Python-style string formatting, as per pattern % tuple Limitations: only %d %s %r %%
   * are supported.
   *
   * @param pattern a format string.
   * @param arguments a tuple containing positional arguments.
   * @return the formatted string.
   */
  public static String formatWithList(String pattern, List<?> arguments) {
    return getPrinter().formatWithList(pattern, arguments).toString();
  }

  /**
   * Perform Python-style string formatting, lazily.
   *
   * @param pattern a format string.
   * @param arguments positional arguments.
   * @return the formatted string.
   */
  public static Formattable formattable(final String pattern, Object... arguments) {
    final ImmutableList<Object> args = ImmutableList.copyOf(arguments);
    return new Formattable() {
      @Override
      public String toString() {
        return formatWithList(pattern, args);
      }

      @Override
      public void formatTo(Formatter formatter, int flags, int width, int precision) {
        Printer.getPrinter(formatter.out()).formatWithList(pattern, args);
      }
    };
  }

  /**
   * Append a char to a buffer. In case of {@link IOException} throw an {@link AssertionError}
   * instead
   *
   * @return buffer
   */
  public static Appendable append(Appendable buffer, char c) {
    try {
      return buffer.append(c);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Append a char sequence to a buffer. In case of {@link IOException} throw an
   * {@link AssertionError} instead
   *
   * @return buffer
   */
  public static Appendable append(Appendable buffer, CharSequence s) {
    try {
      return buffer.append(s);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Append a char sequence range to a buffer. In case of {@link IOException} throw an
   * {@link AssertionError} instead
   * @return buffer
   */
  private static Appendable append(Appendable buffer, CharSequence s, int start, int end) {
    try {
      return buffer.append(s, start, end);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
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

    private static LengthLimitedAppendable create(Appendable original, int limit) {
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
    // TODO(bazel-team): Given an input list
    //
    //     [1, 2, 3, [10, 20, 30, 40, 50, 60], 4, 5, 6]
    //
    // the inner list gets doubly mangled as
    //
    //     [1, 2, 3, [10, 20, 30, 40, <2 more argu...<2 more arguments>], <3 more arguments>]
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
      return this.append(csq.subSequence(start, end));
    }

    @Override
    public Appendable append(char c) throws IOException {
      return this.append(String.valueOf(c));
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

  /** Actual class that implements Printer API */
  public static final class BasePrinter implements SkylarkPrinter {
    // Methods of this class should not recurse through static methods of Printer

    private final Appendable buffer;

    /**
     * Creates a printer instance.
     *
     * @param buffer the Appendable to which to print the representation
     */
    private BasePrinter(Appendable buffer) {
      this.buffer = buffer;
    }

    @Override
    public String toString() {
      return buffer.toString();
    }

    /**
     * Print an informal representation of object x. Currently only differs from repr in the
     * behavior for strings and labels at top-level, that are returned as is rather than quoted.
     *
     * @param o the object
     * @return the buffer, in fluent style
     */
    public BasePrinter str(Object o) {
      if (o instanceof SkylarkPrintableValue) {
        ((SkylarkPrintableValue) o).str(this);
        return this;
      }

      if (o instanceof String) {
        return this.append((String) o);
      }
      return this.repr(o);
    }

    /**
     * Print an official representation of object x. For regular data structures, the value should
     * be parsable back into an equal data structure.
     *
     * @param o the string a representation of which to repr.
     * @return BasePrinter.
     */
    @Override
    public BasePrinter repr(Object o) {
      if (o == null) {
        throw new NullPointerException(); // Java null is not a valid Skylark value.

      } else if (o instanceof SkylarkValue) {
        ((SkylarkValue) o).repr(this);

      } else if (o instanceof String) {
        writeString((String) o);

      } else if (o instanceof Integer || o instanceof Double) {
        this.append(o.toString());

      } else if (o == Boolean.TRUE) {
        this.append("True");

      } else if (o == Boolean.FALSE) {
        this.append("False");

      } else if (o instanceof Map<?, ?>) {
        Map<?, ?> dict = (Map<?, ?>) o;
        this.printList(dict.entrySet(), "{", ", ", "}", null);

      } else if (o instanceof List<?>) {
        List<?> seq = (List<?>) o;
        this.printList(seq, false);

      } else if (o instanceof Map.Entry<?, ?>) {
        Map.Entry<?, ?> entry = (Map.Entry<?, ?>) o;
        this.repr(entry.getKey());
        this.append(": ");
        this.repr(entry.getValue());

      } else if (o instanceof PathFragment) {
        this.append(((PathFragment) o).getPathString());

      } else if (o instanceof Class<?>) {
        this.append(EvalUtils.getDataTypeNameFromClass((Class<?>) o));

      } else if (o instanceof ASTNode || o instanceof Location) {
        // AST node objects and locations are printed in tracebacks and error messages,
        // it's safe to print their toString representations
        this.append(o.toString());

      } else {
        // TODO(bazel-team): change to a special representation for unknown objects
        this.append(o.toString());
      }

      return this;
    }

    /**
     * Write a properly escaped Skylark representation of a string to a buffer.
     *
     * @param s the string a representation of which to repr.
     * @return the Appendable, in fluent style.
     */
    private BasePrinter writeString(String s) {
      this.append(SKYLARK_QUOTATION_MARK);
      int len = s.length();
      for (int i = 0; i < len; i++) {
        char c = s.charAt(i);
        escapeCharacter(c);
      }
      return this.append(SKYLARK_QUOTATION_MARK);
    }

    private BasePrinter backslashChar(char c) {
      return this.append('\\').append(c);
    }

    private BasePrinter escapeCharacter(char c) {
      if (c == SKYLARK_QUOTATION_MARK) {
        return backslashChar(c);
      }
      switch (c) {
        case '\\':
          return backslashChar('\\');
        case '\r':
          return backslashChar('r');
        case '\n':
          return backslashChar('n');
        case '\t':
          return backslashChar('t');
        default:
          if (c < 32) {
            //TODO(bazel-team): support \x escapes
            return this.append(String.format("\\x%02x", (int) c));
          }
          return this.append(c); // no need to support UTF-8
      } // endswitch
    }

    /**
     * Print a list of object representations
     *
     * @param list the list of objects to repr (each as with repr)
     * @param before a string to print before the list items, e.g. an opening bracket
     * @param separator a separator to print between items
     * @param after a string to print after the list items, e.g. a closing bracket
     * @param singletonTerminator null or a string to print after the list if it is a singleton The
     *     singleton case is notably relied upon in python syntax to distinguish a tuple of size one
     *     such as ("foo",) from a merely parenthesized object such as ("foo").
     * @return the BasePrinter.
     */
    @Override
    public BasePrinter printList(
        Iterable<?> list,
        String before,
        String separator,
        String after,
        @Nullable String singletonTerminator) {
      return printAbbreviatedList(list, before, separator, after, singletonTerminator, -1, -1);
    }

    /**
     * Print a list of object representations.
     *
     * <p>The length of the output will be limited when both {@code maxItemsToPrint} and {@code
     * criticalItemsStringLength} have values greater than zero.
     *
     * @param list the list of objects to repr (each as with repr)
     * @param before a string to print before the list
     * @param separator a separator to print between each object
     * @param after a string to print after the list
     * @param singletonTerminator null or a string to print after the list if it is a singleton The
     *     singleton case is notably relied upon in python syntax to distinguish a tuple of size one
     *     such as ("foo",) from a merely parenthesized object such as ("foo").
     * @param maxItemsToPrint the maximum number of elements to be printed.
     * @param criticalItemsStringLength a soft limit for the total string length of all arguments.
     *     'Soft' means that this limit may be exceeded because of formatting.
     * @return the BasePrinter.
     */
    public BasePrinter printAbbreviatedList(
        Iterable<?> list,
        String before,
        String separator,
        String after,
        @Nullable String singletonTerminator,
        int maxItemsToPrint,
        int criticalItemsStringLength) {
      this.append(before);
      int len = 0;
      // Limits the total length of the string representation of the elements, if specified.
      if (maxItemsToPrint > 0 && criticalItemsStringLength > 0) {
        len =
            appendListElements(
                LengthLimitedAppendable.create(buffer, criticalItemsStringLength),
                list,
                separator,
                maxItemsToPrint);
      } else {
        len = appendListElements(list, separator);
      }
      if (singletonTerminator != null && len == 1) {
        this.append(singletonTerminator);
      }
      return this.append(after);
    }

    /**
     * Appends the given elements to the specified {@link Appendable} and returns the number of
     * elements.
     */
    private int appendListElements(Iterable<?> list, String separator) {
      boolean printSeparator = false; // don't print the separator before the first element
      int len = 0;
      for (Object o : list) {
        if (printSeparator) {
          this.append(separator);
        }
        this.repr(o);
        printSeparator = true;
        len++;
      }
      return len;
    }

    /**
     * Tries to append the given elements to the specified {@link Appendable} until specific limits
     * are reached.
     *
     * @return the number of appended elements.
     */
    private int appendListElements(
        LengthLimitedAppendable appendable,
        Iterable<?> list,
        String separator,
        int maxItemsToPrint) {
      boolean printSeparator = false; // don't print the separator before the first element
      boolean skipArgs = false;
      int items = Iterables.size(list);
      int len = 0;
      // We don't want to print "1 more arguments", hence we don't skip arguments if there is only
      // one above the limit.
      int itemsToPrint = (items - maxItemsToPrint == 1) ? items : maxItemsToPrint;
      appendable.enforceLimit();
      for (Object o : list) {
        // We don't want to print "1 more arguments", even if we hit the string limit.
        if (len == itemsToPrint || (appendable.hasHitLimit() && len < items - 1)) {
          skipArgs = true;
          break;
        }
        if (printSeparator) {
          this.append(separator);
        }
        Printer.getPrinter(appendable).repr(o);
        printSeparator = true;
        len++;
      }
      appendable.ignoreLimit();
      if (skipArgs) {
        this.append(separator);
        this.append(String.format("<%d more arguments>", items - len));
      }
      return len;
    }

    /**
     * Print a Skylark list or tuple of object representations
     *
     * @param list the contents of the list or tuple
     * @param isTuple if true the list will be formatted with parentheses and with a trailing comma
     *     in case of one-element tuples.
     * @param maxItemsToPrint the maximum number of elements to be printed.
     * @param criticalItemsStringLength a soft limit for the total string length of all arguments.
     *     'Soft' means that this limit may be exceeded because of formatting.
     * @return the Appendable, in fluent style.
     */
    public BasePrinter printAbbreviatedList(
        Iterable<?> list, boolean isTuple, int maxItemsToPrint, int criticalItemsStringLength) {
      if (isTuple) {
        return this.printAbbreviatedList(list, "(", ", ", ")", ",", 
            maxItemsToPrint, criticalItemsStringLength);
      } else {
        return this.printAbbreviatedList(list, "[", ", ", "]", null, 
            maxItemsToPrint, criticalItemsStringLength);
      }
    }

    @Override
    public BasePrinter printList(Iterable<?> list, boolean isTuple) {
      return this.printAbbreviatedList(list, isTuple, -1, -1);
    }

    /**
     * Perform Python-style string formatting, as per pattern % tuple Limitations: only %d %s %r %%
     * are supported.
     *
     * @param pattern a format string.
     * @param arguments an array containing positional arguments.
     * @return the formatted string.
     */
    @Override
    public BasePrinter format(String pattern, Object... arguments) {
      return this.formatWithList(pattern, ImmutableList.copyOf(arguments));
    }

    /**
     * Perform Python-style string formatting, as per pattern % tuple Limitations: only %d %s %r %%
     * are supported.
     *
     * @param pattern a format string.
     * @param arguments a tuple containing positional arguments.
     * @return the formatted string.
     */
    @Override
    public BasePrinter formatWithList(String pattern, List<?> arguments) {
      // TODO(bazel-team): support formatting arguments, and more complex Python patterns.
      // N.B. MissingFormatWidthException is the only kind of IllegalFormatException
      // whose constructor can take and display arbitrary error message, hence its use below.

      int length = pattern.length();
      int argLength = arguments.size();
      int i = 0; // index of next character in pattern
      int a = 0; // index of next argument in arguments

      while (i < length) {
        int p = pattern.indexOf('%', i);
        if (p == -1) {
          Printer.append(buffer, pattern, i, length);
          break;
        }
        if (p > i) {
          Printer.append(buffer, pattern, i, p);
        }
        if (p == length - 1) {
          throw new MissingFormatWidthException(
              "incomplete format pattern ends with %: " + this.repr(pattern));
        }
        char directive = pattern.charAt(p + 1);
        i = p + 2;
        switch (directive) {
          case '%':
            this.append('%');
            continue;
          case 'd':
          case 'r':
          case 's':
            if (a >= argLength) {
              throw new MissingFormatWidthException(
                  "not enough arguments for format pattern "
                      + this.repr(pattern)
                      + ": "
                      + this.repr(Tuple.copyOf(arguments)));
            }
            Object argument = arguments.get(a++);
            switch (directive) {
              case 'd':
                if (argument instanceof Integer) {
                  this.append(argument.toString());
                  continue;
                } else {
                  throw new MissingFormatWidthException(
                      "invalid argument " + this.repr(argument) + " for format pattern %d");
                }
              case 'r':
                this.repr(argument);
                continue;
              case 's':
                this.str(argument);
                continue;
            }
            // fall through
          default:
            throw new MissingFormatWidthException(
                // The call to Printer.repr doesn't cause an infinite recursion because it's
                // only used to format a string properly
                String.format("unsupported format character \"%s\" at index %s in %s",
                    String.valueOf(directive), p + 1, Printer.repr(pattern)));
        }
      }
      if (a < argLength) {
        throw new MissingFormatWidthException(
            "not all arguments converted during string formatting");
      }
      return this;
    }

    @Override
    public BasePrinter append(char c) {
      Printer.append(buffer, c);
      return this;
    }

    @Override
    public BasePrinter append(CharSequence s) {
      Printer.append(buffer, s);
      return this;
    }
  }
}
