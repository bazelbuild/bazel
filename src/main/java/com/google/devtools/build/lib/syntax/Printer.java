// Copyright 2015 Google Inc. All rights reserved.
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
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Formattable;
import java.util.Formatter;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.Map;
import java.util.MissingFormatWidthException;

/**
 * (Pretty) Printing of Skylark values
 */
public final class Printer {

  private static final char SKYLARK_QUOTATION_MARK = '"';
  
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
  public static Appendable print(Appendable buffer, Object o, char quotationMark) {
    if (o instanceof Label) {
      return append(buffer, o.toString());  // Pretty-print a label like a string
    }
    if (o instanceof String) {
      return append(buffer, (String) o);
    }
    return write(buffer, o, quotationMark);
  }
  
  public static Appendable print(Appendable buffer, Object o) {
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

    } else if (o instanceof String) {
      writeString(buffer, (String) o, quotationMark);

    } else if (o instanceof Integer || o instanceof Double) {
      append(buffer, o.toString());

    } else if (o == Environment.NONE) {
      append(buffer, "None");

    } else if (o == Boolean.TRUE) {
      append(buffer, "True");

    } else if (o == Boolean.FALSE) {
      append(buffer, "False");

    } else if (o instanceof List<?>) {
      List<?> seq = (List<?>) o;
      printList(buffer, seq, EvalUtils.isImmutable(seq), quotationMark);

    } else if (o instanceof SkylarkList) {
      SkylarkList list = (SkylarkList) o;
      printList(buffer, list.toList(), list.isTuple(), quotationMark);

    } else if (o instanceof Map<?, ?>) {
      Map<?, ?> dict = (Map<?, ?>) o;
      printList(buffer, dict.entrySet(), "{", ", ", "}", null, quotationMark);

    } else if (o instanceof Map.Entry<?, ?>) {
      Map.Entry<?, ?> entry = (Map.Entry<?, ?>) o;
      write(buffer, entry.getKey(), quotationMark);
      append(buffer, ": ");
      write(buffer, entry.getValue(), quotationMark);

    } else if (o instanceof SkylarkNestedSet) {
      SkylarkNestedSet set = (SkylarkNestedSet) o;
      append(buffer, "set(");
      printList(buffer, set, "[", ", ", "]", null, quotationMark);
      Order order = set.getOrder();
      if (order != Order.STABLE_ORDER) {
        append(buffer, ", order = \"" + order.getName() + "\"");
      }
      append(buffer, ")");

    } else if (o instanceof BaseFunction) {
      BaseFunction func = (BaseFunction) o;
      append(buffer, "<function " + func.getName() + ">");

    } else if (o instanceof Label) {
      write(buffer, o.toString(), quotationMark);

    } else if (o instanceof FilesetEntry) {
           FilesetEntry entry = (FilesetEntry) o;
      append(buffer, "FilesetEntry(srcdir = ");
      write(buffer, entry.getSrcLabel().toString(), quotationMark);
      append(buffer, ", files = ");
      write(buffer, makeStringList(entry.getFiles()), quotationMark);
      append(buffer, ", excludes = ");
      write(buffer, makeList(entry.getExcludes()), quotationMark);
      append(buffer, ", destdir = ");
      write(buffer, entry.getDestDir().getPathString(), quotationMark);
      append(buffer, ", strip_prefix = ");
      write(buffer, entry.getStripPrefix(), quotationMark);
      append(buffer, ", symlinks = ");
      append(buffer, quotationMark);
      append(buffer, entry.getSymlinkBehavior().toString());
      append(buffer, quotationMark);
      append(buffer, ")");

    } else if (o instanceof PathFragment) {
      append(buffer, ((PathFragment) o).getPathString());

    } else {
      append(buffer, o.toString());
    }

    return buffer;
  }

  public static Appendable write(Appendable buffer, Object o) {
    return write(buffer, o, SKYLARK_QUOTATION_MARK);
  }
  
  // Throughout this file, we transform IOException into AssertionError.
  // During normal operations, we only use in-memory Appendable-s that
  // cannot cause an IOException.
  private static Appendable append(Appendable buffer, char c) {
    try {
      return buffer.append(c);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  private static Appendable append(Appendable buffer, CharSequence s) {
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
  public static Appendable writeString(Appendable buffer, String s, char quote) {
    append(buffer, quote);
    int len = s.length();
    for (int i = 0; i < len; i++) {
      char c = s.charAt(i);
      escapeCharacter(buffer, c, quote);
    }
    return append(buffer, quote);
  }

  /**
   * Write a properly escaped Skylark representation of a string to a buffer.
   * By default, standard Skylark convention is used, i.e., double-quoted single-line string,
   * as opposed to standard Python convention, i.e. single-quoted single-line string.
   *
   * @param buffer the Appendable we're writing to.
   * @param s the string a representation of which to write.
   * @return the buffer, in fluent style.
   */
  public static Appendable writeString(Appendable buffer, String s) {
    return writeString(buffer, s, SKYLARK_QUOTATION_MARK);
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
  public static Appendable printList(Appendable buffer, Iterable<?> list, String before,
      String separator, String after, String singletonTerminator, char quotationMark) {
    boolean printSeparator = false; // don't print the separator before the first element
    int len = 0;
    append(buffer, before);
    for (Object o : list) {
      if (printSeparator) {
        append(buffer, separator);
      }
      write(buffer, o, quotationMark);
      printSeparator = true;
      len++;
    }
    if (singletonTerminator != null && len == 1) {
      append(buffer, singletonTerminator);
    }
    return append(buffer, after);
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
   * @return the Appendable, in fluent style.
   */
  public static Appendable printList(
      Appendable buffer, Iterable<?> list, boolean isTuple, char quotationMark) {
    if (isTuple) {
      return printList(buffer, list, "(", ", ", ")", ",", quotationMark);
    } else {
      return printList(buffer, list, "[", ", ", "]", null, quotationMark);
    }
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

  private static List<?> makeList(Collection<?> list) {
    return list == null ? Lists.newArrayList() : Lists.newArrayList(list);
  }

  private static List<String> makeStringList(List<Label> labels) {
    if (labels == null) {
      return Collections.emptyList();
    }
    List<String> strings = Lists.newArrayListWithCapacity(labels.size());
    for (Label label : labels) {
      strings.add(label.toString());
    }
    return strings;
  }

  /**
   * Convert BUILD language objects to Formattable so JDK can render them correctly.
   * Don't do this for numeric or string types because we want %d, %x, %s to work.
   * This function is intended for use in assertions such as Precondition.checkArgument
   * so that you only pay the cost of computing the string when the assertion passes.
   */
  public static Object strFormattable(final Object o, final char quotationMark) {
    if (o instanceof Integer || o instanceof Double || o instanceof String) {
      return o;
    } else {
      return new Formattable() {
        @Override
        public String toString() {
          return str(o, quotationMark);
        }

        @Override
        public void formatTo(Formatter formatter, int flags, int width, int precision) {
          print(formatter.out(), o, quotationMark);
        }
      };
    }
  }

  public static Object strFormattable(final Object o) {
    return strFormattable(o, SKYLARK_QUOTATION_MARK);
  }
  
  /**
   * Convert BUILD language objects to Formattable so JDK can render them correctly.
   * Don't do this for numeric or string types because we want %d, %x, %s to work.
   * This function is intended for use in assertions such as Precondition.checkArgument
   * so that you only pay the cost of computing the string when the assertion passes.
   */
  public static Object reprFormattable(final Object o, final char quotationMark) {
    if (o instanceof Integer || o instanceof Double) {
      return o;
    } else {
      return new Formattable() {
        @Override
        public String toString() {
          return repr(o);
        }

        @Override
        public void formatTo(Formatter formatter, int flags, int width, int precision) {
          write(formatter.out(), o, quotationMark);
        }
      };
    }
  }

  public static Object reprFormattable(final Object o) {
    return reprFormattable(o, SKYLARK_QUOTATION_MARK);
  }
  
  
  /*
   * N.B. MissingFormatWidthException is the only kind of IllegalFormatException
   * whose constructor can take and display arbitrary error message, hence its use below.
   */

  /**
   * Perform Python-style string formatting.
   *
   * @param pattern a format string.
   * @param arguments a tuple containing positional arguments.
   * @return the formatted string.
   */
  public static String formatString(String pattern, List<?> arguments)
      throws IllegalFormatException {
    return format(new StringBuilder(), pattern, arguments).toString();
  }

  /**
   * Perform Python-style string formatting.
   *
   * @param pattern a format string.
   * @param arguments positional arguments.
   * @return the formatted string.
   */
  public static String format(String pattern, Object... arguments) throws IllegalFormatException {
    return formatString(pattern, ImmutableList.copyOf(arguments));
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
  public static Appendable format(Appendable buffer, String pattern, List<?> arguments)
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
                + repr(SkylarkList.tuple(arguments)));
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
}
