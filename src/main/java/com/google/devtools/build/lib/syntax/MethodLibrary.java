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

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.syntax.EvalUtils.ComparisonException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkDocumentationCategory;
import net.starlark.java.annot.StarlarkGlobalLibrary;
import net.starlark.java.annot.StarlarkMethod;

/** The universal predeclared functions of core Starlark. */
@StarlarkGlobalLibrary
class MethodLibrary {

  @StarlarkMethod(
      name = "min",
      doc =
          "Returns the smallest one of all given arguments. "
              + "If only one argument is provided, it must be a non-empty iterable. "
              + "It is an error if elements are not comparable (for example int with string). "
              + "<pre class=\"language-python\">min(2, 5, 4) == 2\n"
              + "min([5, 6, 3]) == 3</pre>",
      extraPositionals = @Param(name = "args", doc = "The elements to be checked."))
  public Object min(Sequence<?> args) throws EvalException {
    try {
      return findExtreme(args, EvalUtils.STARLARK_COMPARATOR.reverse());
    } catch (ComparisonException e) {
      throw new EvalException(null, e);
    }
  }

  @StarlarkMethod(
      name = "max",
      doc =
          "Returns the largest one of all given arguments. "
              + "If only one argument is provided, it must be a non-empty iterable."
              + "It is an error if elements are not comparable (for example int with string). "
              + "<pre class=\"language-python\">max(2, 5, 4) == 5\n"
              + "max([5, 6, 3]) == 6</pre>",
      extraPositionals = @Param(name = "args", doc = "The elements to be checked."))
  public Object max(Sequence<?> args) throws EvalException {
    try {
      return findExtreme(args, EvalUtils.STARLARK_COMPARATOR);
    } catch (ComparisonException e) {
      throw new EvalException(null, e);
    }
  }

  /** Returns the maximum element from this list, as determined by maxOrdering. */
  private static Object findExtreme(Sequence<?> args, Ordering<Object> maxOrdering)
      throws EvalException {
    // Args can either be a list of items to compare, or a singleton list whose element is an
    // iterable of items to compare. In either case, there must be at least one item to compare.
    try {
      Iterable<?> items = (args.size() == 1) ? Starlark.toIterable(args.get(0)) : args;
      return maxOrdering.max(items);
    } catch (NoSuchElementException ex) {
      throw new EvalException(null, "expected at least one item", ex);
    }
  }

  @StarlarkMethod(
      name = "all",
      doc =
          "Returns true if all elements evaluate to True or if the collection is empty. "
              + "Elements are converted to boolean using the <a href=\"#bool\">bool</a> function."
              + "<pre class=\"language-python\">all([\"hello\", 3, True]) == True\n"
              + "all([-1, 0, 1]) == False</pre>",
      parameters = {
        @Param(
            name = "elements",
            type = Object.class,
            noneable = true,
            doc = "A string or a collection of elements.")
      })
  public Boolean all(Object collection) throws EvalException {
    return !hasElementWithBooleanValue(collection, false);
  }

  @StarlarkMethod(
      name = "any",
      doc =
          "Returns true if at least one element evaluates to True. "
              + "Elements are converted to boolean using the <a href=\"#bool\">bool</a> function."
              + "<pre class=\"language-python\">any([-1, 0, 1]) == True\n"
              + "any([False, 0, \"\"]) == False</pre>",
      parameters = {
        @Param(
            name = "elements",
            type = Object.class,
            noneable = true,
            doc = "A string or a collection of elements.")
      })
  public Boolean any(Object collection) throws EvalException {
    return hasElementWithBooleanValue(collection, true);
  }

  private static boolean hasElementWithBooleanValue(Object seq, boolean value)
      throws EvalException {
    for (Object x : Starlark.toIterable(seq)) {
      if (Starlark.truth(x) == value) {
        return true;
      }
    }
    return false;
  }

  @StarlarkMethod(
      name = "sorted",
      doc =
          "Sort a collection. Elements should all belong to the same orderable type, they are "
              + "sorted by their value (in ascending order). "
              + "It is an error if elements are not comparable (for example int with string)."
              + "<pre class=\"language-python\">sorted([3, 5, 4]) == [3, 4, 5]</pre>",
      parameters = {
        @Param(name = "iterable", type = Object.class, doc = "The iterable sequence to sort."),
        @Param(
            name = "key",
            doc = "An optional function applied to each element before comparison.",
            named = true,
            defaultValue = "None",
            positional = false,
            noneable = true),
        @Param(
            name = "reverse",
            type = Boolean.class,
            doc = "Return results in descending order.",
            named = true,
            defaultValue = "False",
            positional = false)
      },
      useStarlarkThread = true)
  public StarlarkList<?> sorted(
      Object iterable, final Object key, Boolean reverse, final StarlarkThread thread)
      throws EvalException, InterruptedException {
    Object[] array = Starlark.toArray(iterable);
    if (key == Starlark.NONE) {
      try {
        Arrays.sort(array, EvalUtils.STARLARK_COMPARATOR);
      } catch (EvalUtils.ComparisonException e) {
        throw Starlark.errorf("%s", e.getMessage());
      }
    } else if (key instanceof StarlarkCallable) {
      final StarlarkCallable keyfn = (StarlarkCallable) key;

      class KeyComparator implements Comparator<Object> {
        Exception e;

        @Override
        public int compare(Object x, Object y) {
          try {
            return EvalUtils.STARLARK_COMPARATOR.compare(callKeyFunc(x), callKeyFunc(y));
          } catch (InterruptedException | EvalException e) {
            if (this.e == null) {
              this.e = e;
            }
            return 0;
          }
        }

        Object callKeyFunc(Object x) throws EvalException, InterruptedException {
          return Starlark.call(thread, keyfn, Collections.singletonList(x), ImmutableMap.of());
        }
      }

      KeyComparator comp = new KeyComparator();
      try {
        Arrays.sort(array, comp);
      } catch (EvalUtils.ComparisonException e) {
        throw Starlark.errorf("%s", e.getMessage());
      }

      if (comp.e != null) {
        if (comp.e instanceof InterruptedException) {
          throw (InterruptedException) comp.e;
        }
        throw (EvalException) comp.e;
      }
    } else {
      throw Starlark.errorf("%s object is not callable", Starlark.repr(Starlark.type(key)));
    }

    if (reverse) {
      reverse(array);
    }
    return StarlarkList.wrap(thread.mutability(), array);
  }

  private static void reverse(Object[] array) {
    for (int i = 0, j = array.length - 1; i < j; i++, j--) {
      Object tmp = array[i];
      array[i] = array[j];
      array[j] = tmp;
    }
  }

  @StarlarkMethod(
      name = "reversed",
      doc =
          "Returns a list that contains the elements of the original sequence in reversed order."
              + "<pre class=\"language-python\">reversed([3, 5, 4]) == [4, 5, 3]</pre>",
      parameters = {
        @Param(
            name = "sequence",
            type = Sequence.class,
            doc = "The sequence (list or tuple) to be reversed."),
      },
      useStarlarkThread = true)
  public StarlarkList<?> reversed(Sequence<?> sequence, StarlarkThread thread)
      throws EvalException {
    Object[] array = Starlark.toArray(sequence);
    reverse(array);
    return StarlarkList.wrap(thread.mutability(), array);
  }

  @StarlarkMethod(
      name = "tuple",
      doc =
          "Returns a tuple with the same elements as the given iterable value."
              + "<pre class=\"language-python\">tuple([1, 2]) == (1, 2)\n"
              + "tuple((2, 3, 2)) == (2, 3, 2)\n"
              + "tuple({5: \"a\", 2: \"b\", 4: \"c\"}) == (5, 2, 4)</pre>",
      parameters = {@Param(name = "x", defaultValue = "()", doc = "The object to convert.")})
  public Tuple<?> tuple(Object x) throws EvalException {
    if (x instanceof Tuple) {
      return (Tuple<?>) x;
    }
    return Tuple.wrap(Starlark.toArray(x));
  }

  @StarlarkMethod(
      name = "list",
      doc =
          "Returns a new list with the same elements as the given iterable value."
              + "<pre class=\"language-python\">list([1, 2]) == [1, 2]\n"
              + "list((2, 3, 2)) == [2, 3, 2]\n"
              + "list({5: \"a\", 2: \"b\", 4: \"c\"}) == [5, 2, 4]</pre>",
      parameters = {@Param(name = "x", defaultValue = "[]", doc = "The object to convert.")},
      useStarlarkThread = true)
  public StarlarkList<?> list(Object x, StarlarkThread thread) throws EvalException {
    return StarlarkList.wrap(thread.mutability(), Starlark.toArray(x));
  }

  @StarlarkMethod(
      name = "len",
      doc =
          "Returns the length of a string, sequence (such as a list or tuple), dict, or other"
              + " iterable.",
      parameters = {@Param(name = "x", doc = "The value whose length to report.")},
      useStarlarkThread = true)
  public Integer len(Object x, StarlarkThread thread) throws EvalException {
    int len = Starlark.len(x);
    if (len < 0) {
      throw Starlark.errorf("%s is not iterable", Starlark.type(x));
    }
    return len;
  }

  @StarlarkMethod(
      name = "str",
      doc =
          "Converts any object to string. This is useful for debugging."
              + "<pre class=\"language-python\">str(\"ab\") == \"ab\"\n"
              + "str(8) == \"8\"</pre>",
      parameters = {@Param(name = "x", doc = "The object to convert.", noneable = true)})
  public String str(Object x) throws EvalException {
    return Starlark.str(x);
  }

  @StarlarkMethod(
      name = "repr",
      doc =
          "Converts any object to a string representation. This is useful for debugging.<br>"
              + "<pre class=\"language-python\">repr(\"ab\") == '\"ab\"'</pre>",
      parameters = {@Param(name = "x", doc = "The object to convert.", noneable = true)})
  public String repr(Object x) {
    return Starlark.repr(x);
  }

  @StarlarkMethod(
      name = "bool",
      doc =
          "Constructor for the bool type. "
              + "It returns <code>False</code> if the object is <code>None</code>, <code>False"
              + "</code>, an empty string (<code>\"\"</code>), the number <code>0</code>, or an "
              + "empty collection (e.g. <code>()</code>, <code>[]</code>). "
              + "Otherwise, it returns <code>True</code>.",
      parameters = {
        @Param(
            name = "x",
            defaultValue = "False",
            doc = "The variable to convert.",
            noneable = true)
      })
  public Boolean bool(Object x) throws EvalException {
    return Starlark.truth(x);
  }

  private final ImmutableMap<String, Integer> intPrefixes =
      ImmutableMap.of("0b", 2, "0o", 8, "0x", 16);

  @StarlarkMethod(
      name = "int",
      doc =
          "Returns x as an int value."
              + "<ul>"
              + "<li>If <code>x</code> is already an int, it is returned as-is."
              + "<li>If <code>x</code> is a boolean, a true value returns 1 and a false value "
              + "    returns 0."
              + "<li>If <code>x</code> is a string, it must have the format "
              + "    <code>&lt;sign&gt;&lt;prefix&gt;&lt;digits&gt;</code>. "
              + "    <code>&lt;sign&gt;</code> is either <code>\"+\"</code>, <code>\"-\"</code>, "
              + "    or empty (interpreted as positive). <code>&lt;digits&gt;</code> are a "
              + "    sequence of digits from 0 up to <code>base</code> - 1, where the letters a-z "
              + "    (or equivalently, A-Z) are used as digits for 10-35. In the case where "
              + "    <code>base</code> is 2/8/16, <code>&lt;prefix&gt;</code> is optional and may "
              + "    be 0b/0o/0x (or equivalently, 0B/0O/0X) respectively; if the "
              + "    <code>base</code> is any other value besides these bases or the special value "
              + "    0, the prefix must be empty. In the case where <code>base</code> is 0, the "
              + "    string is interpreted as an integer literal, in the sense that one of the "
              + "    bases 2/8/10/16 is chosen depending on which prefix if any is used. If "
              + "    <code>base</code> is 0, no prefix is used, and there is more than one digit, "
              + "    the leading digit cannot be 0; this is to avoid confusion between octal and "
              + "    decimal. The magnitude of the number represented by the string must be within "
              + "    the allowed range for the int type."
              + "</ul>"
              + "This function fails if <code>x</code> is any other type, or if the value is a "
              + "string not satisfying the above format. Unlike Python's <code>int()</code> "
              + "function, this function does not allow zero arguments, and does not allow "
              + "extraneous whitespace for string arguments."
              + "<p>Examples:"
              + "<pre class=\"language-python\">"
              + "int(\"123\") == 123\n"
              + "int(\"-123\") == -123\n"
              + "int(\"+123\") == 123\n"
              + "int(\"FF\", 16) == 255\n"
              + "int(\"0xFF\", 16) == 255\n"
              + "int(\"10\", 0) == 10\n"
              + "int(\"-0x10\", 0) == -16"
              + "</pre>",
      parameters = {
        @Param(name = "x", type = Object.class, doc = "The string to convert."),
        @Param(
            name = "base",
            type = Object.class,
            defaultValue = "unbound",
            doc =
                "The base used to interpret a string value; defaults to 10. Must be between 2 "
                    + "and 36 (inclusive), or 0 to detect the base as if <code>x</code> were an "
                    + "integer literal. This parameter must not be supplied if the value is not a "
                    + "string.",
            named = true)
      })
  public Integer convertToInt(Object x, Object base) throws EvalException {
    if (x instanceof String) {
      if (base == Starlark.UNBOUND) {
        base = 10;
      } else if (!(base instanceof Integer)) {
        throw Starlark.errorf("base must be an integer (got '%s')", Starlark.type(base));
      }
      return fromString((String) x, (Integer) base);
    } else {
      if (base != Starlark.UNBOUND) {
        throw Starlark.errorf("int() can't convert non-string with explicit base");
      }
      if (x instanceof Boolean) {
        return ((Boolean) x).booleanValue() ? 1 : 0;
      } else if (x instanceof Integer) {
        return (Integer) x;
      }
      throw Starlark.errorf("%s is not of type string or int or bool", Starlark.repr(x));
    }
  }

  private int fromString(String string, int base) throws EvalException {
    String stringForErrors = string;

    boolean isNegative = false;
    if (string.isEmpty()) {
      throw Starlark.errorf("string argument to int() cannot be empty");
    }
    char c = string.charAt(0);
    if (c == '+') {
      string = string.substring(1);
    } else if (c == '-') {
      string = string.substring(1);
      isNegative = true;
    }

    String prefix = getIntegerPrefix(string);
    String digits;
    if (prefix == null) {
      // Nothing to strip. Infer base 10 if autodetection was requested (base == 0).
      digits = string;
      if (base == 0) {
        if (string.length() > 1 && string.startsWith("0")) {
          // We don't infer the base when input starts with '0' (due
          // to confusion between octal and decimal).
          throw Starlark.errorf(
              "cannot infer base for int() when value begins with a 0: %s",
              Starlark.repr(stringForErrors));
        }
        base = 10;
      }
    } else {
      // Strip prefix. Infer base from prefix if unknown (base == 0), or else verify its
      // consistency.
      digits = string.substring(prefix.length());
      int expectedBase = intPrefixes.get(prefix);
      if (base == 0) {
        base = expectedBase;
      } else if (base != expectedBase) {
        throw Starlark.errorf(
            "invalid literal for int() with base %d: %s", base, Starlark.repr(stringForErrors));
      }
    }

    if (base < 2 || base > 36) {
      throw Starlark.errorf("int() base must be >= 2 and <= 36");
    }
    try {
      // Negate by prepending a negative symbol, rather than by using arithmetic on the
      // result, to handle the edge case of -2^31 correctly.
      String parseable = isNegative ? "-" + digits : digits;
      return Integer.parseInt(parseable, base);
    } catch (NumberFormatException | ArithmeticException e) {
      throw new EvalException(
          null,
          Starlark.format("invalid literal for int() with base %d: %r", base, stringForErrors),
          e);
    }
  }

  @Nullable
  private String getIntegerPrefix(String value) {
    value = Ascii.toLowerCase(value);
    for (String prefix : intPrefixes.keySet()) {
      if (value.startsWith(prefix)) {
        return prefix;
      }
    }
    return null;
  }

  @StarlarkMethod(
      name = "dict",
      doc =
          "Creates a <a href=\"dict.html\">dictionary</a> from an optional positional "
              + "argument and an optional set of keyword arguments. In the case where the same key "
              + "is given multiple times, the last value will be used. Entries supplied via "
              + "keyword arguments are considered to come after entries supplied via the "
              + "positional argument.",
      parameters = {
        @Param(
            name = "args",
            type = Object.class,
            defaultValue = "[]",
            doc =
                "Either a dictionary or a list of entries. Entries must be tuples or lists with "
                    + "exactly two elements: key, value."),
      },
      extraKeywords = @Param(name = "kwargs", doc = "Dictionary of additional entries."),
      useStarlarkThread = true)
  public Dict<?, ?> dict(Object args, Dict<String, Object> kwargs, StarlarkThread thread)
      throws EvalException {
    Dict<?, ?> dict =
        args instanceof Dict
            ? (Dict) args
            : Dict.getDictFromArgs("dict", args, thread.mutability());
    return Dict.plus(dict, kwargs, thread.mutability());
  }

  @StarlarkMethod(
      name = "enumerate",
      doc =
          "Returns a list of pairs (two-element tuples), with the index (int) and the item from"
              + " the input sequence.\n<pre class=\"language-python\">"
              + "enumerate([24, 21, 84]) == [(0, 24), (1, 21), (2, 84)]</pre>\n",
      parameters = {
        // Note Python uses 'sequence' keyword instead of 'list'. We may want to change tihs
        // some day.
        @Param(name = "list", type = Object.class, doc = "input sequence.", named = true),
        @Param(
            name = "start",
            type = Integer.class,
            doc = "start index.",
            defaultValue = "0",
            named = true)
      },
      useStarlarkThread = true)
  public StarlarkList<?> enumerate(Object input, Integer start, StarlarkThread thread)
      throws EvalException {
    Object[] array = Starlark.toArray(input);
    for (int i = 0; i < array.length; i++) {
      array[i] = Tuple.pair(i + start, array[i]); // update in place
    }
    return StarlarkList.wrap(thread.mutability(), array);
  }

  @StarlarkMethod(
      name = "hash",
      doc =
          "Return a hash value for a string. This is computed deterministically using the same "
              + "algorithm as Java's <code>String.hashCode()</code>, namely: "
              + "<pre class=\"language-python\">s[0] * (31^(n-1)) + s[1] * (31^(n-2)) + ... + "
              + "s[n-1]</pre> Hashing of values besides strings is not currently supported.",
      // Deterministic hashing is important for the consistency of builds, hence why we
      // promise a specific algorithm. This is in contrast to Java (Object.hashCode()) and
      // Python, which promise stable hashing only within a given execution of the program.
      parameters = {@Param(name = "value", type = String.class, doc = "String value to hash.")})
  public Integer hash(String value) throws EvalException {
    return value.hashCode();
  }

  @StarlarkMethod(
      name = "range",
      doc =
          "Creates a list where items go from <code>start</code> to <code>stop</code>, using a "
              + "<code>step</code> increment. If a single argument is provided, items will "
              + "range from 0 to that element."
              + "<pre class=\"language-python\">range(4) == [0, 1, 2, 3]\n"
              + "range(3, 9, 2) == [3, 5, 7]\n"
              + "range(3, 0, -1) == [3, 2, 1]</pre>",
      parameters = {
        @Param(
            name = "start_or_stop",
            type = Integer.class,
            doc =
                "Value of the start element if stop is provided, "
                    + "otherwise value of stop and the actual start is 0"),
        @Param(
            name = "stop_or_none",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc =
                "optional index of the first item <i>not</i> to be included in the resulting "
                    + "list; generation of the list stops before <code>stop</code> is reached."),
        @Param(
            name = "step",
            type = Integer.class,
            defaultValue = "1",
            doc = "The increment (default is 1). It may be negative.")
      },
      useStarlarkThread = true)
  public Sequence<Integer> range(
      Integer startOrStop, Object stopOrNone, Integer step, StarlarkThread thread)
      throws EvalException {
    int start;
    int stop;
    if (stopOrNone == Starlark.NONE) {
      start = 0;
      stop = startOrStop;
    } else if (stopOrNone instanceof Integer) {
      start = startOrStop;
      stop = (Integer) stopOrNone;
    } else {
      throw Starlark.errorf("want int, got %s", Starlark.type(stopOrNone));
    }
    if (step == 0) {
      throw Starlark.errorf("step cannot be 0");
    }
    return new RangeList(start, stop, step);
  }

  /** Returns true if the object has a field of the given name, otherwise false. */
  @StarlarkMethod(
      name = "hasattr",
      doc =
          "Returns True if the object <code>x</code> has an attribute or method of the given "
              + "<code>name</code>, otherwise False. Example:<br>"
              + "<pre class=\"language-python\">hasattr(ctx.attr, \"myattr\")</pre>",
      parameters = {
        @Param(name = "x", doc = "The object to check.", noneable = true),
        @Param(name = "name", type = String.class, doc = "The name of the attribute.")
      },
      useStarlarkThread = true)
  public Boolean hasattr(Object obj, String name, StarlarkThread thread) throws EvalException {
    return Starlark.hasattr(thread.getSemantics(), obj, name);
  }

  @StarlarkMethod(
      name = "getattr",
      doc =
          "Returns the struct's field of the given name if it exists. If not, it either returns "
              + "<code>default</code> (if specified) or raises an error. "
              + "<code>getattr(x, \"foobar\")</code> is equivalent to <code>x.foobar</code>."
              + "<pre class=\"language-python\">getattr(ctx.attr, \"myattr\")\n"
              + "getattr(ctx.attr, \"myattr\", \"mydefault\")</pre>",
      parameters = {
        @Param(name = "x", doc = "The struct whose attribute is accessed.", noneable = true),
        @Param(name = "name", doc = "The name of the struct attribute."),
        @Param(
            name = "default",
            defaultValue = "unbound",
            doc =
                "The default value to return in case the struct "
                    + "doesn't have an attribute of the given name.",
            noneable = true)
      },
      useStarlarkThread = true)
  public Object getattr(Object obj, String name, Object defaultValue, StarlarkThread thread)
      throws EvalException, InterruptedException {
    return Starlark.getattr(
        thread.mutability(),
        thread.getSemantics(),
        obj,
        name,
        defaultValue == Starlark.UNBOUND ? null : defaultValue);
  }

  @StarlarkMethod(
      name = "dir",
      doc =
          "Returns a list of strings: the names of the attributes and "
              + "methods of the parameter object.",
      parameters = {@Param(name = "x", doc = "The object to check.", noneable = true)},
      useStarlarkThread = true)
  public StarlarkList<?> dir(Object object, StarlarkThread thread) throws EvalException {
    return Starlark.dir(thread.mutability(), thread.getSemantics(), object);
  }

  @StarlarkMethod(
      name = "fail",
      doc =
          "Raises an error that cannot be intercepted. It can be used anywhere, "
              + "both in the loading phase and in the analysis phase.",
      parameters = {
        @Param(
            name = "msg",
            type = Object.class,
            doc = "Error to display for the user. The object is converted to a string.",
            defaultValue = "None",
            named = true,
            noneable = true),
        @Param(
            name = "attr",
            type = String.class,
            noneable = true,
            defaultValue = "None",
            doc =
                "The name of the attribute that caused the error. This is used only for "
                    + "error reporting.",
            named = true)
      })
  public NoneType fail(Object msg, Object attr) throws EvalException {
    String str = Starlark.str(msg);
    if (attr != Starlark.NONE) {
      str = Starlark.format("attribute %s: %s", attr, str);
    }
    throw Starlark.errorf("%s", str);
  }

  @StarlarkMethod(
      name = "print",
      doc =
          "Prints <code>args</code> as debug output. It will be prefixed with the string <code>"
              + "\"DEBUG\"</code> and the location (file and line number) of this call. The "
              + "exact way in which the arguments are converted to strings is unspecified and may "
              + "change at any time. In particular, it may be different from (and more detailed "
              + "than) the formatting done by <a href='#str'><code>str()</code></a> and <a "
              + "href='#repr'><code>repr()</code></a>."
              + "<p>Using <code>print</code> in production code is discouraged due to the spam it "
              + "creates for users. For deprecations, prefer a hard error using <a href=\"#fail\">"
              + "<code>fail()</code></a> whenever possible.",
      parameters = {
        @Param(
            name = "sep",
            type = String.class,
            defaultValue = "\" \"",
            named = true,
            positional = false,
            doc = "The separator string between the objects, default is space (\" \").")
      },
      // NB: as compared to Python3, we're missing optional named-only arguments 'end' and 'file'
      extraPositionals = @Param(name = "args", doc = "The objects to print."),
      useStarlarkThread = true)
  public NoneType print(String sep, Sequence<?> args, StarlarkThread thread) throws EvalException {
    Printer p = Printer.getPrinter();
    String separator = "";
    for (Object x : args) {
      p.append(separator);
      p.debugPrint(x);
      separator = sep;
    }
    // As part of the integration test "starlark_flag_test.sh", if the
    // "--internal_starlark_flag_test_canary" flag is enabled, append an extra marker string to
    // the output.
    if (thread.getSemantics().internalStarlarkFlagTestCanary()) {
      p.append("<== Starlark flag test ==>");
    }

    thread.getPrintHandler().print(thread, p.toString());
    return Starlark.NONE;
  }

  @StarlarkMethod(
      name = "type",
      doc =
          "Returns the type name of its argument. This is useful for debugging and "
              + "type-checking. Examples:"
              + "<pre class=\"language-python\">"
              + "type(2) == \"int\"\n"
              + "type([1]) == \"list\"\n"
              + "type(struct(a = 2)) == \"struct\""
              + "</pre>"
              + "This function might change in the future. To write Python-compatible code and "
              + "be future-proof, use it only to compare return values: "
              + "<pre class=\"language-python\">"
              + "if type(x) == type([]):  # if x is a list"
              + "</pre>",
      parameters = {@Param(name = "x", doc = "The object to check type of.", noneable = true)})
  public String type(Object object) {
    // There is no 'type' type in Starlark, so we return a string with the type name.
    return Starlark.type(object);
  }

  @StarlarkMethod(
      name = "zip",
      doc =
          "Returns a <code>list</code> of <code>tuple</code>s, where the i-th tuple contains "
              + "the i-th element from each of the argument sequences or iterables. The list has "
              + "the size of the shortest input. With a single iterable argument, it returns a "
              + "list of 1-tuples. With no arguments, it returns an empty list. Examples:"
              + "<pre class=\"language-python\">"
              + "zip()  # == []\n"
              + "zip([1, 2])  # == [(1,), (2,)]\n"
              + "zip([1, 2], [3, 4])  # == [(1, 3), (2, 4)]\n"
              + "zip([1, 2], [3, 4, 5])  # == [(1, 3), (2, 4)]</pre>",
      extraPositionals = @Param(name = "args", doc = "lists to zip."),
      useStarlarkThread = true)
  public StarlarkList<?> zip(Sequence<?> args, StarlarkThread thread) throws EvalException {
    Iterator<?>[] iterators = new Iterator<?>[args.size()];
    for (int i = 0; i < args.size(); i++) {
      iterators[i] = Starlark.toIterable(args.get(i)).iterator();
    }
    ArrayList<Tuple<?>> result = new ArrayList<>();
    boolean allHasNext;
    do {
      allHasNext = !args.isEmpty();
      List<Object> elem = Lists.newArrayListWithExpectedSize(args.size());
      for (Iterator<?> iterator : iterators) {
        if (iterator.hasNext()) {
          elem.add(iterator.next());
        } else {
          allHasNext = false;
        }
      }
      if (allHasNext) {
        result.add(Tuple.copyOf(elem));
      }
    } while (allHasNext);
    return StarlarkList.copyOf(thread.mutability(), result);
  }

  /** Starlark int type. */
  @StarlarkBuiltin(
      name = "int",
      category = StarlarkDocumentationCategory.BUILTIN,
      doc =
          "A type to represent integers. It can represent any number between -2147483648 and "
              + "2147483647 (included). "
              + "Examples of int values:<br>"
              + "<pre class=\"language-python\">"
              + "153\n"
              + "0x2A  # hexadecimal literal\n"
              + "0o54  # octal literal\n"
              + "23 * 2 + 5\n"
              + "100 / -7\n"
              + "100 % -7  # -5 (unlike in some other languages)\n"
              + "int(\"18\")\n"
              + "</pre>")
  static final class IntModule implements StarlarkValue {} // (documentation only)

  /** Starlark bool type. */
  @StarlarkBuiltin(
      name = "bool",
      category = StarlarkDocumentationCategory.BUILTIN,
      doc =
          "A type to represent booleans. There are only two possible values: "
              + "<a href=\"globals.html#True\">True</a> and "
              + "<a href=\"globals.html#False\">False</a>. "
              + "Any value can be converted to a boolean using the "
              + "<a href=\"globals.html#bool\">bool</a> function.")
  static final class BoolModule implements StarlarkValue {} // (documentation only)
}
