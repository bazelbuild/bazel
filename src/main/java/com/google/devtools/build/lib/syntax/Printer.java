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

import com.google.common.base.Strings;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import java.io.IOException;
import java.util.Arrays;
import java.util.Formattable;
import java.util.Formatter;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.Map;
import java.util.MissingFormatWidthException;
import java.util.UnknownFormatConversionException;
import javax.annotation.Nullable;

/** (Pretty) Printing of Skylark values */
public class Printer {

  /**
   * Creates an instance of {@link BasePrinter} that wraps an existing buffer.
   *
   * @param buffer an {@link Appendable}
   * @return new {@link BasePrinter}
   */
  static BasePrinter getPrinter(Appendable buffer) {
    return new BasePrinter(buffer);
  }

  /**
   * Creates an instance of {@link BasePrinter} with an empty buffer.
   *
   * @return new {@link BasePrinter}
   */
  public static BasePrinter getPrinter() {
    return new BasePrinter(new StringBuilder());
  }

  /**
   * Creates an instance of {@link PrettyPrinter} with an empty buffer.
   *
   * @return new {@link PrettyPrinter}
   */
  public static PrettyPrinter getPrettyPrinter() {
    return new PrettyPrinter(new StringBuilder());
  }

  /**
   * Creates an instance of {@link BasePrinter} with an empty buffer and whose format strings allow
   * only %s and %%.
   */
  public static BasePrinter getSimplifiedPrinter() {
    return new BasePrinter(new StringBuilder(), /*simplifiedFormatStrings=*/ true);
  }

  private Printer() {}

  // These static methods proxy to the similar methods of BasePrinter

  /** Format an object with Skylark's {@code debugPrint}. */
  static String debugPrint(Object x) {
    return getPrinter().debugPrint(x).toString();
  }

  /**
   * Perform Python-style string formatting, lazily.
   *
   * @param pattern a format string.
   * @param arguments positional arguments.
   * @return the formatted string.
   */
  static Formattable formattable(final String pattern, Object... arguments) {
    final List<Object> args = Arrays.asList(arguments);
    return new Formattable() {
      @Override
      public String toString() {
        return Starlark.formatWithList(pattern, args);
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
  private static Appendable append(Appendable buffer, char c) {
    try {
      return buffer.append(c);
    } catch (IOException e) {
      throw new AssertionError(e);
    }
  }

  /**
   * Append a char sequence to a buffer. In case of {@link IOException} throw an {@link
   * AssertionError} instead
   *
   * @return buffer
   */
  private static Appendable append(Appendable buffer, CharSequence s) {
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

  /** Actual class that implements Printer API */
  public static class BasePrinter implements SkylarkPrinter {
    // Methods of this class should not recurse through static methods of Printer

    protected final Appendable buffer;

    /**
     * If true, the only percent sequences allowed in format strings are %s substitutions and %%
     * escapes.
     */
    protected final boolean simplifiedFormatStrings;

    /**
     * Creates a printer.
     *
     * @param buffer the {@link Appendable} that will be written to
     * @param simplifiedFormatStrings if true, format strings will allow only %s and %%
     */
    protected BasePrinter(Appendable buffer, boolean simplifiedFormatStrings) {
      this.buffer = buffer;
      this.simplifiedFormatStrings = simplifiedFormatStrings;
    }

    /**
     * Creates a printer that writes to the given buffer and that does not use simplified format
     * strings.
     */
    protected BasePrinter(Appendable buffer) {
      this(buffer, /*simplifiedFormatStrings=*/ false);
    }

    /**
     * Creates a printer that uses a fresh buffer and that does not use simplified format strings.
     */
    protected BasePrinter() {
      this(new StringBuilder());
    }

    @Override
    public String toString() {
      return buffer.toString();
    }

    /**
     * Print an informal debug-only representation of object x.
     *
     * @param o the object
     * @return the buffer, in fluent style
     */
    public BasePrinter debugPrint(Object o) {
      if (o instanceof SkylarkValue) {
        ((SkylarkValue) o).debugPrint(this);
        return this;
      }

      return this.str(o);
    }

    /**
     * Prints the informal representation of value {@code o}. Unlike {@code repr(x)}, it does not
     * quote strings at top level, though strings and other values appearing as elements of other
     * structures are quoted as if by {@code repr}.
     *
     * <p>Implementations of SkylarkValue may define their own behavior of {@code str}.
     */
    public BasePrinter str(Object o) {
      if (o instanceof SkylarkValue) {
        ((SkylarkValue) o).str(this);
        return this;
      }

      if (o instanceof String) {
        return this.append((String) o);
      }
      return this.repr(o);
    }

    /**
     * Prints the quoted representation of Starlark value {@code o}. The quoted form is often a
     * Starlark expression that evaluates to {@code o}.
     *
     * <p>Implementations of SkylarkValue may define their own behavior of {@code repr}.
     *
     * <p>In addition to Starlark values, {@code repr} also prints instances of classes Map, List,
     * Map.Entry, Class, Node, or Location. To avoid nondeterminism, all other values are printed
     * opaquely.
     */
    @Override
    public BasePrinter repr(Object o) {
      if (o == null) {
        // Java null is not a valid Skylark value, but sometimes printers are used on non-Skylark
        // values such as Locations or ASTs.
        this.append("null");

      } else if (o instanceof SkylarkValue) {
        ((SkylarkValue) o).repr(this);

      } else if (o instanceof String) {
        writeString((String) o);

      } else if (o instanceof Integer || o instanceof Double) {
        this.append(o.toString());

      } else if (Boolean.TRUE.equals(o)) {
        this.append("True");

      } else if (Boolean.FALSE.equals(o)) {
        this.append("False");

        // -- non-Starlark values --

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

      } else if (o instanceof Class<?>) {
        this.append(EvalUtils.getDataTypeNameFromClass((Class<?>) o));

      } else if (o instanceof Node || o instanceof Location) {
        // AST node objects and locations are printed in tracebacks and error messages,
        // it's safe to print their toString representations
        this.append(o.toString());

      } else {
        // For now, we print all unknown values opaquely.
        // Historically this was a defense against accidental nondeterminism,
        // but Starlark code cannot access values of o that would reach here,
        // and native code is already trusted to be deterministic.
        // TODO(adonovan): replace this with a default behavior of this.append(o),
        // once we require that all @Skylark-annotated classes implement SkylarkValue.
        // (After all, Java code can call String.format, which also calls toString.)
        this.append("<unknown object " + o.getClass().getName() + ">");
      }

      return this;
    }

    /**
     * Write a properly escaped Skylark representation of a string to a buffer.
     *
     * @param s the string a representation of which to repr.
     * @return this printer.
     */
    protected BasePrinter writeString(String s) {
      this.append('"');
      int len = s.length();
      for (int i = 0; i < len; i++) {
        char c = s.charAt(i);
        escapeCharacter(c);
      }
      return this.append('"');
    }

    private BasePrinter backslashChar(char c) {
      return this.append('\\').append(c);
    }

    private BasePrinter escapeCharacter(char c) {
      if (c == '"') {
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
     * @return this printer.
     */
    @Override
    public BasePrinter printList(
        Iterable<?> list,
        String before,
        String separator,
        String after,
        @Nullable String singletonTerminator) {

      this.append(before);
      int len = appendListElements(list, separator);
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
     * Print a Skylark list or tuple of object representations
     *
     * @param list the contents of the list or tuple
     * @param isTuple if true the list will be formatted with parentheses and with a trailing comma
     *     in case of one-element tuples. 'Soft' means that this limit may be exceeded because of
     *     formatting.
     * @return this printer.
     */
    @Override
    public BasePrinter printList(Iterable<?> list, boolean isTuple) {
      if (isTuple) {
        return this.printList(list, "(", ", ", ")", ",");
      } else {
        return this.printList(list, "[", ", ", "]", null);
      }
    }

    /**
     * Perform Python-style string formatting, similar to the {@code pattern % tuple} syntax.
     *
     * <p>The only supported placeholder patterns are
     * <ul>
     *   <li>{@code %s} (convert as if by {@code str()})
     *   <li>{@code %r} (convert as if by {@code repr()})
     *   <li>{@code %d} (convert an integer to its decimal representation)
     * </ul>
     * To encode a literal percent character, escape it as {@code %%}. It is an error to have a
     * non-escaped {@code %} at the end of the string or followed by any character not listed above.
     *
     * <p>If this printer has {@code simplifiedFormatStrings} set, only {@code %s} and {@code %%}
     * are permitted.
     *
     * @param pattern a format string that may contain placeholders
     * @param arguments an array containing arguments to substitute into the placeholders in order
     * @return the formatted string
     * @throws IllegalFormatException if {@code pattern} is not a valid format string, or if
     *     {@code arguments} mismatches the number or type of placeholders in {@code pattern}
     */
    @Override
    public BasePrinter format(String pattern, Object... arguments) {
      return this.formatWithList(pattern, Arrays.asList(arguments));
    }

    /**
     * Perform Python-style string formatting, similar to the {@code pattern % tuple} syntax.
     *
     * <p>Same as {@link #format(String, Object...)}, but with a list instead of variadic args.
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
            if (simplifiedFormatStrings && (directive != 's')) {
              throw new UnknownFormatConversionException(
                  "cannot use %" + directive + " substitution placeholder when "
                      + "simplifiedFormatStrings is set");
            }
            if (a >= argLength) {
              throw new MissingFormatWidthException(
                  "not enough arguments for format pattern "
                      + Starlark.repr(pattern)
                      + ": "
                      + Starlark.repr(Tuple.copyOf(arguments)));
            }
            Object argument = arguments.get(a++);
            switch (directive) {
              case 'd':
                if (argument instanceof Integer) {
                  this.append(argument.toString());
                  continue;
                } else {
                  throw new MissingFormatWidthException(
                      "invalid argument " + Starlark.repr(argument) + " for format pattern %d");
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
                // The call to Starlark.repr doesn't cause an infinite recursion because it's
                // only used to format a string properly
                String.format(
                    "unsupported format character \"%s\" at index %s in %s",
                    String.valueOf(directive), p + 1, Starlark.repr(pattern)));
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

    BasePrinter append(CharSequence sequence, int start, int end) {
      return this.append(sequence.subSequence(start, end));
    }
  }

  /** A printer that breaks lines between the entries of lists, with proper indenting. */
  public static class PrettyPrinter extends BasePrinter {
    static final int BASE_INDENT = 4;
    private int indent;

    protected PrettyPrinter(Appendable buffer) {
      super(buffer);
      indent = 0;
    }

    @Override
    public BasePrinter printList(
        Iterable<?> list,
        String before,
        String untrimmedSeparator,
        String after,
        @Nullable String singletonTerminator) {

      // If the list is empty, do not split the presentation over
      // several lines.
      if (!list.iterator().hasNext()) {
        this.append(before + after);
        return this;
      }

      String separator = untrimmedSeparator.trim();

      this.append(before + "\n");
      indent += BASE_INDENT;
      boolean printSeparator = false; // don't print the separator before the first element
      int len = 0;
      for (Object o : list) {
        if (printSeparator) {
          this.append(separator + "\n");
        }
        this.append(Strings.repeat(" ", indent));
        this.repr(o);
        printSeparator = true;
        len++;
      }
      if (singletonTerminator != null && len == 1) {
        this.append(singletonTerminator);
      }
      this.append("\n");
      indent -= BASE_INDENT;
      this.append(Strings.repeat(" ", indent) + after);
      return this;
    }
  }
}
