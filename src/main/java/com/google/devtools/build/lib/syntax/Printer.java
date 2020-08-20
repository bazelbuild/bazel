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

import java.util.Arrays;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.Map;
import java.util.MissingFormatWidthException;

/**
 * A printer of Starlark values.
 *
 * <p>Subclasses may override methods such as {@link #repr} and {@link #printList} to alter the
 * formatting behavior.
 */
// TODO(adonovan): disallow printing of objects that are not Starlark values.
public class Printer {

  private final StringBuilder buffer;

  /** Creates a printer that writes to the given buffer. */
  public Printer(StringBuilder buffer) {
    this.buffer = buffer;
  }

  /** Creates a printer that uses a fresh buffer. */
  public Printer() {
    this(new StringBuilder());
  }

  /** Appends a char to the printer's buffer */
  public final Printer append(char c) {
    buffer.append(c);
    return this;
  }

  /** Appends a char sequence to the printer's buffer */
  public final Printer append(CharSequence s) {
    buffer.append(s);
    return this;
  }

  /** Appends a char subsequence to the printer's buffer */
  public final Printer append(CharSequence s, int start, int end) {
    buffer.append(s, start, end);
    return this;
  }

  /**
   * Appends a list to the printer's buffer. List elements are rendered with {@code repr}.
   *
   * <p>May be overridden by subclasses.
   *
   * @param list the list of objects to repr (each as with repr)
   * @param before a string to print before the list items, e.g. an opening bracket
   * @param separator a separator to print between items
   * @param after a string to print after the list items, e.g. a closing bracket
   */
  public Printer printList(Iterable<?> list, String before, String separator, String after) {
    this.append(before);
    String sep = "";
    for (Object elem : list) {
      this.append(sep);
      sep = separator;
      this.repr(elem);
    }
    return this.append(after);
  }

  @Override
  public final String toString() {
    return buffer.toString();
  }

  /**
   * Appends the {@code StarlarkValue.debugPrint} representation of a value (as used by the Starlark
   * {@code print} statement) to the printer's buffer.
   *
   * <p>Implementations of StarlarkValue may define their own behavior of {@code debugPrint}.
   */
  public Printer debugPrint(Object o) {
    if (o instanceof StarlarkValue) {
      ((StarlarkValue) o).debugPrint(this);
      return this;
    }

    return this.str(o);
  }

  /**
   * Appends the {@code StarlarkValue.str} representation of a value to the printer's buffer. Unlike
   * {@code repr(x)}, it does not quote strings at top level, though strings and other values
   * appearing as elements of other structures are quoted as if by {@code repr}.
   *
   * <p>Implementations of StarlarkValue may define their own behavior of {@code str}.
   */
  public Printer str(Object o) {
    if (o instanceof StarlarkValue) {
      ((StarlarkValue) o).str(this);
      return this;

    } else if (o instanceof String) {
      return this.append((String) o);

    } else {
      return this.repr(o);
    }
  }

  /**
   * Appends the {@code StarlarkValue.repr} (quoted) representation of a value to the printer's
   * buffer. The quoted form is often a Starlark expression that evaluates to the value.
   *
   * <p>Implementations of StarlarkValue may define their own behavior of {@code repr}.
   *
   * <p>In addition to Starlark values, {@code repr} also prints instances of classes Map, List,
   * Map.Entry, or Class. All other values are formatted using their {@code toString} method.
   * TODO(adonovan): disallow that.
   */
  public Printer repr(Object o) {
    if (o == null) {
      // Java null is not a valid Starlark value, but sometimes printers are used on non-Starlark
      // values such as Locations or Nodes.
      this.append("null");

    } else if (o instanceof StarlarkValue) {
      ((StarlarkValue) o).repr(this);

    } else if (o instanceof String) {
      appendQuoted((String) o);

    } else if (o instanceof Integer) {
      this.buffer.append((int) o);

    } else if (o instanceof Boolean) {
      this.append(((boolean) o) ? "True" : "False");

      // -- non-Starlark values --

    } else if (o instanceof Map) {
      Map<?, ?> dict = (Map<?, ?>) o;
      this.printList(dict.entrySet(), "{", ", ", "}");

    } else if (o instanceof List) {
      this.printList((List) o, "[", ", ", "]");

    } else if (o instanceof Map.Entry) {
      Map.Entry<?, ?> entry = (Map.Entry<?, ?>) o;
      this.repr(entry.getKey());
      this.append(": ");
      this.repr(entry.getValue());

    } else if (o instanceof Class) {
      this.append(Starlark.classType((Class<?>) o));

    } else {
      // All other non-Starlark Java values (e.g. Node, Location).
      // Starlark code cannot access values of o that would reach here,
      // and native code is already trusted to be deterministic.
      this.append(o.toString());
    }

    return this;
  }

  private Printer appendQuoted(String s) {
    this.append('"');
    int len = s.length();
    for (int i = 0; i < len; i++) {
      char c = s.charAt(i);
      escapeCharacter(c);
    }
    return this.append('"');
  }

  private Printer backslashChar(char c) {
    return this.append('\\').append(c);
  }

  private Printer escapeCharacter(char c) {
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
          // TODO(bazel-team): support \x escapes
          return this.append(String.format("\\x%02x", (int) c));
        }
        return this.append(c); // no need to support UTF-8
    }
  }

  @Deprecated // kept briefly to avoid breaking Copybara. Remedy: inline it.
  public final void format(String pattern, Object... arguments) {
    Printer.format(this, pattern, arguments);
  }

  /**
   * Appends a string, formatted as if by Starlark's {@code str % tuple} operator, to the printer's
   * buffer.
   *
   * <p>Supported conversions:
   *
   * <ul>
   *   <li>{@code %s} (convert as if by {@code str()})
   *   <li>{@code %r} (convert as if by {@code repr()})
   *   <li>{@code %d} (convert an integer to its decimal representation)
   * </ul>
   *
   * To encode a literal percent character, escape it as {@code %%}. It is an error to have a
   * non-escaped {@code %} at the end of the string or followed by any character not listed above.
   *
   * @param format the format string
   * @param arguments an array containing arguments to substitute into the format operators in order
   * @throws IllegalFormatException if the format string is invalid or the arguments do not match it
   */
  public static void format(Printer printer, String format, Object... arguments) {
    formatWithList(printer, format, Arrays.asList(arguments));
  }

  /** Same as {@link #format}, but with a list instead of variadic args. */
  public static void formatWithList(Printer printer, String pattern, List<?> arguments) {
    // N.B. MissingFormatWidthException is the only kind of IllegalFormatException
    // whose constructor can take and display arbitrary error message, hence its use below.
    // TODO(adonovan): this suggests we're using the wrong exception. Throw IAE?

    int length = pattern.length();
    int argLength = arguments.size();
    int i = 0; // index of next character in pattern
    int a = 0; // index of next argument in arguments

    while (i < length) {
      int p = pattern.indexOf('%', i);
      if (p == -1) {
        printer.append(pattern, i, length);
        break;
      }
      if (p > i) {
        printer.append(pattern, i, p);
      }
      if (p == length - 1) {
        throw new MissingFormatWidthException(
            "incomplete format pattern ends with %: " + Starlark.repr(pattern));
      }
      char directive = pattern.charAt(p + 1);
      i = p + 2;
      switch (directive) {
        case '%':
          printer.append('%');
          continue;
        case 'd':
        case 'r':
        case 's':
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
              if (!(argument instanceof Integer)) {
                throw new MissingFormatWidthException(
                    "invalid argument " + Starlark.repr(argument) + " for format pattern %d");
              }
              printer.append(argument.toString());
              continue;

            case 'r':
              printer.repr(argument);
              continue;

            case 's':
              printer.str(argument);
              continue;

            default:
              // no-op
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
      throw new MissingFormatWidthException("not all arguments converted during string formatting");
    }
  }
}
