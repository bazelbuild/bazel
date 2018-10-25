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

import static java.util.stream.Collectors.joining;

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkSignature;
import com.google.devtools.build.lib.syntax.EvalUtils.ComparisonException;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.Type.ConversionException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.TreeSet;
import javax.annotation.Nullable;

/** A helper class containing built in functions for the Skylark language. */
public class MethodLibrary {

  private MethodLibrary() {}

  @SkylarkSignature(
      name = "min",
      returnType = Object.class,
      doc =
          "Returns the smallest one of all given arguments. "
              + "If only one argument is provided, it must be a non-empty iterable. "
              + "It is an error if elements are not comparable (for example int with string). "
              + "<pre class=\"language-python\">min(2, 5, 4) == 2\n"
              + "min([5, 6, 3]) == 3</pre>",
      extraPositionals =
          @Param(name = "args", type = SkylarkList.class, doc = "The elements to be checked."),
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction min =
      new BuiltinFunction("min") {
        @SuppressWarnings("unused") // Accessed via Reflection.
        public Object invoke(SkylarkList<?> args, Location loc, Environment env)
            throws EvalException {
          try {
            return findExtreme(args, EvalUtils.SKYLARK_COMPARATOR.reverse(), loc, env);
          } catch (ComparisonException e) {
            throw new EvalException(loc, e);
          }
        }
      };

  @SkylarkSignature(
      name = "max",
      returnType = Object.class,
      doc =
          "Returns the largest one of all given arguments. "
              + "If only one argument is provided, it must be a non-empty iterable."
              + "It is an error if elements are not comparable (for example int with string). "
              + "<pre class=\"language-python\">max(2, 5, 4) == 5\n"
              + "max([5, 6, 3]) == 6</pre>",
      extraPositionals =
          @Param(name = "args", type = SkylarkList.class, doc = "The elements to be checked."),
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction max =
      new BuiltinFunction("max") {
        @SuppressWarnings("unused") // Accessed via Reflection.
        public Object invoke(SkylarkList<?> args, Location loc, Environment env)
            throws EvalException {
          try {
            return findExtreme(args, EvalUtils.SKYLARK_COMPARATOR, loc, env);
          } catch (ComparisonException e) {
            throw new EvalException(loc, e);
          }
        }
      };

  /** Returns the maximum element from this list, as determined by maxOrdering. */
  private static Object findExtreme(
      SkylarkList<?> args, Ordering<Object> maxOrdering, Location loc, Environment env)
      throws EvalException {
    // Args can either be a list of items to compare, or a singleton list whose element is an
    // iterable of items to compare. In either case, there must be at least one item to compare.
    try {
      Iterable<?> items = (args.size() == 1) ? EvalUtils.toIterable(args.get(0), loc, env) : args;
      return maxOrdering.max(items);
    } catch (NoSuchElementException ex) {
      throw new EvalException(loc, "expected at least one item");
    }
  }

  @SkylarkSignature(
      name = "all",
      returnType = Boolean.class,
      doc =
          "Returns true if all elements evaluate to True or if the collection is empty. "
              + "Elements are converted to boolean using the <a href=\"#bool\">bool</a> function."
              + "<pre class=\"language-python\">all([\"hello\", 3, True]) == True\n"
              + "all([-1, 0, 1]) == False</pre>",
      parameters = {
        @Param(
            name = "elements",
            type = Object.class,
            doc = "A string or a collection of elements.")
      },
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction all =
      new BuiltinFunction("all") {
        @SuppressWarnings("unused") // Accessed via Reflection.
        public Boolean invoke(Object collection, Location loc, Environment env)
            throws EvalException {
          return !hasElementWithBooleanValue(collection, false, loc, env);
        }
      };

  @SkylarkSignature(
      name = "any",
      returnType = Boolean.class,
      doc =
          "Returns true if at least one element evaluates to True. "
              + "Elements are converted to boolean using the <a href=\"#bool\">bool</a> function."
              + "<pre class=\"language-python\">any([-1, 0, 1]) == True\n"
              + "any([False, 0, \"\"]) == False</pre>",
      parameters = {
        @Param(
            name = "elements",
            type = Object.class,
            doc = "A string or a collection of elements.")
      },
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction any =
      new BuiltinFunction("any") {
        @SuppressWarnings("unused") // Accessed via Reflection.
        public Boolean invoke(Object collection, Location loc, Environment env)
            throws EvalException {
          return hasElementWithBooleanValue(collection, true, loc, env);
        }
      };

  private static boolean hasElementWithBooleanValue(
      Object collection, boolean value, Location loc, Environment env) throws EvalException {
    Iterable<?> iterable = EvalUtils.toIterable(collection, loc, env);
    for (Object obj : iterable) {
      if (EvalUtils.toBoolean(obj) == value) {
        return true;
      }
    }
    return false;
  }

  // supported list methods
  @SkylarkSignature(
      name = "sorted",
      returnType = MutableList.class,
      doc =
          "Sort a collection. Elements should all belong to the same orderable type, they are "
              + "sorted by their value (in ascending order). "
              + "It is an error if elements are not comparable (for example int with string)."
              + "<pre class=\"language-python\">sorted([3, 5, 4]) == [3, 4, 5]</pre>",
      parameters = {@Param(name = "self", type = Object.class, doc = "This collection.")},
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction sorted =
      new BuiltinFunction("sorted") {
        public MutableList<?> invoke(Object self, Location loc, Environment env)
            throws EvalException {
          try {
            return MutableList.copyOf(
                env,
                EvalUtils.SKYLARK_COMPARATOR.sortedCopy(EvalUtils.toCollection(self, loc, env)));
          } catch (EvalUtils.ComparisonException e) {
            throw new EvalException(loc, e);
          }
        }
      };

  @SkylarkSignature(
      name = "reversed",
      returnType = MutableList.class,
      doc =
          "Returns a list that contains the elements of the original sequence in reversed order."
              + "<pre class=\"language-python\">reversed([3, 5, 4]) == [4, 5, 3]</pre>",
      parameters = {
        @Param(
            name = "sequence",
            type = Object.class,
            doc = "The sequence to be reversed (string, list or tuple).")
      },
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction reversed =
      new BuiltinFunction("reversed") {
        @SuppressWarnings("unused") // Accessed via Reflection.
        public MutableList<?> invoke(Object sequence, Location loc, Environment env)
            throws EvalException {
          // We only allow lists and strings.
          if (sequence instanceof SkylarkDict) {
            throw new EvalException(
                loc, "Argument to reversed() must be a sequence, not a dictionary.");
          } else if (sequence instanceof NestedSet || sequence instanceof SkylarkNestedSet) {
            throw new EvalException(
                loc, "Argument to reversed() must be a sequence, not a depset.");
          }
          ArrayDeque<Object> tmpList = new ArrayDeque<>();
          for (Object element : EvalUtils.toIterable(sequence, loc, env)) {
            tmpList.addFirst(element);
          }
          return MutableList.copyOf(env, tmpList);
        }
      };

  @SkylarkSignature(
      name = "tuple",
      returnType = Tuple.class,
      doc =
          "Converts a collection (e.g. list, tuple or dictionary) to a tuple."
              + "<pre class=\"language-python\">tuple([1, 2]) == (1, 2)\n"
              + "tuple((2, 3, 2)) == (2, 3, 2)\n"
              + "tuple({5: \"a\", 2: \"b\", 4: \"c\"}) == (5, 2, 4)</pre>",
      parameters = {@Param(name = "x", doc = "The object to convert.")},
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction tuple =
      new BuiltinFunction("tuple") {
        public Tuple<?> invoke(Object x, Location loc, Environment env) throws EvalException {
          return Tuple.copyOf(EvalUtils.toCollection(x, loc, env));
        }
      };

  @SkylarkSignature(
      name = "list",
      returnType = MutableList.class,
      doc =
          "Converts a collection (e.g. list, tuple or dictionary) to a list."
              + "<pre class=\"language-python\">list([1, 2]) == [1, 2]\n"
              + "list((2, 3, 2)) == [2, 3, 2]\n"
              + "list({5: \"a\", 2: \"b\", 4: \"c\"}) == [5, 2, 4]</pre>",
      parameters = {@Param(name = "x", doc = "The object to convert.")},
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction list =
      new BuiltinFunction("list") {
        public MutableList<?> invoke(Object x, Location loc, Environment env) throws EvalException {
          return MutableList.copyOf(env, EvalUtils.toCollection(x, loc, env));
        }
      };

  @SkylarkSignature(
      name = "len",
      returnType = Integer.class,
      doc = "Returns the length of a string, list, tuple, depset, or dictionary.",
      parameters = {@Param(name = "x", doc = "The object to check length of.")},
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction len =
      new BuiltinFunction("len") {
        public Integer invoke(Object x, Location loc, Environment env) throws EvalException {
          if (env.getSemantics().incompatibleDepsetIsNotIterable()
              && x instanceof SkylarkNestedSet) {
            throw new EvalException(
                loc,
                EvalUtils.getDataTypeName(x)
                    + " is not iterable. You may use `len(<depset>.to_list())` instead. Use "
                    + "--incompatible_depset_is_not_iterable=false to temporarily disable this "
                    + "check.");
          }
          int l = EvalUtils.size(x);
          if (l == -1) {
            throw new EvalException(loc, EvalUtils.getDataTypeName(x) + " is not iterable");
          }
          return l;
        }
      };

  @SkylarkSignature(
      name = "str",
      returnType = String.class,
      doc =
          "Converts any object to string. This is useful for debugging."
              + "<pre class=\"language-python\">str(\"ab\") == \"ab\"\n"
              + "str(8) == \"8\"</pre>",
      parameters = {@Param(name = "x", doc = "The object to convert.")})
  private static final BuiltinFunction str =
      new BuiltinFunction("str") {
        public String invoke(Object x) {
          return Printer.str(x);
        }
      };

  @SkylarkSignature(
      name = "repr",
      returnType = String.class,
      doc =
          "Converts any object to a string representation. This is useful for debugging.<br>"
              + "<pre class=\"language-python\">repr(\"ab\") == '\"ab\"'</pre>",
      parameters = {@Param(name = "x", doc = "The object to convert.")})
  private static final BuiltinFunction repr =
      new BuiltinFunction("repr") {
        public String invoke(Object x) {
          return Printer.repr(x);
        }
      };

  @SkylarkSignature(
      name = "bool",
      returnType = Boolean.class,
      doc =
          "Constructor for the bool type. "
              + "It returns <code>False</code> if the object is <code>None</code>, <code>False"
              + "</code>, an empty string (<code>\"\"</code>), the number <code>0</code>, or an "
              + "empty collection (e.g. <code>()</code>, <code>[]</code>). "
              + "Otherwise, it returns <code>True</code>.",
      parameters = {@Param(name = "x", doc = "The variable to convert.")})
  private static final BuiltinFunction bool =
      new BuiltinFunction("bool") {
        public Boolean invoke(Object x) throws EvalException {
          return EvalUtils.toBoolean(x);
        }
      };

  @SkylarkSignature(
      name = "int",
      returnType = Integer.class,
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
                    + "string.")
      },
      useLocation = true)
  private static final BuiltinFunction int_ =
      new BuiltinFunction("int") {
        private final ImmutableMap<String, Integer> intPrefixes =
            ImmutableMap.of("0b", 2, "0o", 8, "0x", 16);

        @SuppressWarnings("unused")
        public Integer invoke(Object x, Object base, Location loc) throws EvalException {
          if (x instanceof String) {
            if (base == Runtime.UNBOUND) {
              base = 10;
            } else if (!(base instanceof Integer)) {
              throw new EvalException(
                  loc, "base must be an integer (got '" + EvalUtils.getDataTypeName(base) + "')");
            }
            return fromString((String) x, loc, (Integer) base);
          } else {
            if (base != Runtime.UNBOUND) {
              throw new EvalException(loc, "int() can't convert non-string with explicit base");
            }
            if (x instanceof Boolean) {
              return ((Boolean) x).booleanValue() ? 1 : 0;
            } else if (x instanceof Integer) {
              return (Integer) x;
            }
            throw new EvalException(
                loc, Printer.format("%r is not of type string or int or bool", x));
          }
        }

        private int fromString(String string, Location loc, int base) throws EvalException {
          String stringForErrors = string;

          boolean isNegative = false;
          if (string.isEmpty()) {
            throw new EvalException(
                loc, Printer.format("string argument to int() cannot be empty"));
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
                throw new EvalException(
                    loc,
                    Printer.format(
                        "cannot infer base for int() when value begins with a 0: %r",
                        stringForErrors));
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
              throw new EvalException(
                  loc,
                  Printer.format(
                      "invalid literal for int() with base %d: %r", base, stringForErrors));
            }
          }

          if (base < 2 || base > 36) {
            throw new EvalException(loc, "int() base must be >= 2 and <= 36");
          }
          try {
            // Negate by prepending a negative symbol, rather than by using arithmetic on the
            // result, to handle the edge case of -2^31 correctly.
            String parseable = isNegative ? "-" + digits : digits;
            return Integer.parseInt(parseable, base);
          } catch (NumberFormatException | ArithmeticException e) {
            throw new EvalException(
                loc,
                Printer.format(
                    "invalid literal for int() with base %d: %r", base, stringForErrors));
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
      };

  @SkylarkSignature(
      name = "dict",
      returnType = SkylarkDict.class,
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
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction dict =
      new BuiltinFunction("dict") {
        public SkylarkDict<?, ?> invoke(
            Object args, SkylarkDict<String, Object> kwargs, Location loc, Environment env)
            throws EvalException {
          SkylarkDict<?, ?> argsDict =
              (args instanceof SkylarkDict)
                  ? (SkylarkDict<?, ?>) args
                  : getDictFromArgs(args, loc, env);
          return SkylarkDict.plus(argsDict, kwargs, env);
        }

        private SkylarkDict<Object, Object> getDictFromArgs(
            Object args, Location loc, Environment env) throws EvalException {
          SkylarkDict<Object, Object> result = SkylarkDict.of(env);
          int pos = 0;
          for (Object element : Type.OBJECT_LIST.convert(args, "parameter args in dict()")) {
            List<Object> pair = convertToPair(element, pos, loc);
            result.put(pair.get(0), pair.get(1), loc, env);
            ++pos;
          }
          return result;
        }

        private List<Object> convertToPair(Object element, int pos, Location loc)
            throws EvalException {
          try {
            List<Object> tuple = Type.OBJECT_LIST.convert(element, "");
            int numElements = tuple.size();
            if (numElements != 2) {
              throw new EvalException(
                  location,
                  String.format(
                      "item #%d has length %d, but exactly two elements are required",
                      pos, numElements));
            }
            return tuple;
          } catch (ConversionException e) {
            throw new EvalException(
                loc, String.format("cannot convert item #%d to a sequence", pos));
          }
        }
      };

  @SkylarkSignature(
      name = "enumerate",
      returnType = MutableList.class,
      doc =
          "Returns a list of pairs (two-element tuples), with the index (int) and the item from"
              + " the input list.\n<pre class=\"language-python\">"
              + "enumerate([24, 21, 84]) == [(0, 24), (1, 21), (2, 84)]</pre>\n",
      parameters = {@Param(name = "list", type = SkylarkList.class, doc = "input list.")},
      useEnvironment = true)
  private static final BuiltinFunction enumerate =
      new BuiltinFunction("enumerate") {
        public MutableList<?> invoke(SkylarkList<?> input, Environment env) throws EvalException {
          int count = 0;
          ArrayList<SkylarkList<?>> result = new ArrayList<>(input.size());
          for (Object obj : input) {
            result.add(Tuple.of(count, obj));
            count++;
          }
          return MutableList.wrapUnsafe(env, result);
        }
      };

  @SkylarkSignature(
      name = "hash",
      returnType = Integer.class,
      doc =
          "Return a hash value for a string. This is computed deterministically using the same "
              + "algorithm as Java's <code>String.hashCode()</code>, namely: "
              + "<pre class=\"language-python\">s[0] * (31^(n-1)) + s[1] * (31^(n-2)) + ... + "
              + "s[n-1]</pre> Hashing of values besides strings is not currently supported.",
      // Deterministic hashing is important for the consistency of builds, hence why we
      // promise a specific algorithm. This is in contrast to Java (Object.hashCode()) and
      // Python, which promise stable hashing only within a given execution of the program.
      parameters = {@Param(name = "value", type = String.class, doc = "String value to hash.")})
  private static final BuiltinFunction hash =
      new BuiltinFunction("hash") {
        public Integer invoke(String value) throws EvalException {
          return value.hashCode();
        }
      };

  @SkylarkSignature(
      name = "range",
      returnType = SkylarkList.class,
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
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction range =
      new BuiltinFunction("range") {
        public SkylarkList<Integer> invoke(
            Integer startOrStop, Object stopOrNone, Integer step, Location loc, Environment env)
            throws EvalException {
          int start;
          int stop;
          if (stopOrNone == Runtime.NONE) {
            start = 0;
            stop = startOrStop;
          } else {
            start = startOrStop;
            stop = Type.INTEGER.convert(stopOrNone, "'stop' operand of 'range'");
          }
          if (step == 0) {
            throw new EvalException(loc, "step cannot be 0");
          }
          RangeList range = RangeList.of(start, stop, step);
          return env.getSemantics().incompatibleRangeType() ? range : range.toMutableList(env);
        }
      };

  /** Returns true if the object has a field of the given name, otherwise false. */
  @SkylarkSignature(
      name = "hasattr",
      returnType = Boolean.class,
      doc =
          "Returns True if the object <code>x</code> has an attribute or method of the given "
              + "<code>name</code>, otherwise False. Example:<br>"
              + "<pre class=\"language-python\">hasattr(ctx.attr, \"myattr\")</pre>",
      parameters = {
        @Param(name = "x", doc = "The object to check."),
        @Param(name = "name", type = String.class, doc = "The name of the attribute.")
      },
      useEnvironment = true)
  private static final BuiltinFunction hasattr =
      new BuiltinFunction("hasattr") {
        @SuppressWarnings("unused")
        public Boolean invoke(Object obj, String name, Environment env) throws EvalException {
          if (obj instanceof ClassObject && ((ClassObject) obj).getValue(name) != null) {
            return true;
          }
          // shouldn't this filter things with struct_field = false?
          return DotExpression.hasMethod(env.getSemantics(), obj, name);
        }
      };

  @SkylarkSignature(
      name = "getattr",
      doc =
          "Returns the struct's field of the given name if it exists. If not, it either returns "
              + "<code>default</code> (if specified) or raises an error. Built-in methods cannot "
              + "currently be retrieved in this way; doing so will result in an error if a "
              + "<code>default</code> is not given. <code>getattr(x, \"foobar\")</code> is "
              + "equivalent to <code>x.foobar</code>."
              + "<pre class=\"language-python\">getattr(ctx.attr, \"myattr\")\n"
              + "getattr(ctx.attr, \"myattr\", \"mydefault\")</pre>",
      parameters = {
        @Param(name = "x", doc = "The struct whose attribute is accessed."),
        @Param(name = "name", doc = "The name of the struct attribute."),
        @Param(
            name = "default",
            defaultValue = "unbound",
            doc =
                "The default value to return in case the struct "
                    + "doesn't have an attribute of the given name.")
      },
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction getattr =
      new BuiltinFunction("getattr") {
        @SuppressWarnings("unused")
        public Object invoke(
            Object obj, String name, Object defaultValue, Location loc, Environment env)
            throws EvalException, InterruptedException {
          Object result = DotExpression.eval(obj, name, loc, env);
          if (result == null) {
            if (defaultValue != Runtime.UNBOUND) {
              return defaultValue;
            }
            throw DotExpression.getMissingFieldException(
                obj, name, loc, env.getSemantics(), "attribute");
          }
          return result;
        }
      };

  @SkylarkSignature(
      name = "dir",
      returnType = MutableList.class,
      doc =
          "Returns a list of strings: the names of the attributes and "
              + "methods of the parameter object.",
      parameters = {@Param(name = "x", doc = "The object to check.")},
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction dir =
      new BuiltinFunction("dir") {
        public MutableList<?> invoke(Object object, Location loc, Environment env)
            throws EvalException {
          // Order the fields alphabetically.
          Set<String> fields = new TreeSet<>();
          if (object instanceof ClassObject) {
            fields.addAll(((ClassObject) object).getFieldNames());
          }
          fields.addAll(Runtime.getBuiltinRegistry().getFunctionNames(object.getClass()));
          fields.addAll(FuncallExpression.getMethodNames(env.getSemantics(), object.getClass()));
          return MutableList.copyOf(env, fields);
        }
      };

  @SkylarkSignature(
      name = "fail",
      doc =
          "Raises an error that cannot be intercepted. It can be used anywhere, "
              + "both in the loading phase and in the analysis phase.",
      returnType = Runtime.NoneType.class,
      parameters = {
        @Param(
            name = "msg",
            type = Object.class,
            doc = "Error to display for the user. The object is converted to a string."),
        @Param(
            name = "attr",
            type = String.class,
            noneable = true,
            defaultValue = "None",
            doc =
                "The name of the attribute that caused the error. This is used only for "
                    + "error reporting.")
      },
      useLocation = true)
  private static final BuiltinFunction fail =
      new BuiltinFunction("fail") {
        public Runtime.NoneType invoke(Object msg, Object attr, Location loc) throws EvalException {
          String str = Printer.str(msg);
          if (attr != Runtime.NONE) {
            str = String.format("attribute %s: %s", attr, str);
          }
          throw new EvalException(loc, str);
        }
      };

  @SkylarkSignature(
      name = "print",
      returnType = Runtime.NoneType.class,
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
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction print =
      new BuiltinFunction("print") {
        public Runtime.NoneType invoke(
            String sep, SkylarkList<?> starargs, Location loc, Environment env)
            throws EvalException {
          String msg = starargs.stream().map(Printer::debugPrint).collect(joining(sep));
          // As part of the integration test "skylark_flag_test.sh", if the
          // "--internal_skylark_flag_test_canary" flag is enabled, append an extra marker string to
          // the output.
          if (env.getSemantics().internalSkylarkFlagTestCanary()) {
            msg += "<== skylark flag test ==>";
          }
          env.handleEvent(Event.debug(loc, msg));
          return Runtime.NONE;
        }
      };

  @SkylarkSignature(
      name = "type",
      returnType = String.class,
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
      parameters = {@Param(name = "x", doc = "The object to check type of.")})
  private static final BuiltinFunction type =
      new BuiltinFunction("type") {
        public String invoke(Object object) {
          // There is no 'type' type in Skylark, so we return a string with the type name.
          return EvalUtils.getDataTypeName(object, false);
        }
      };

  @SkylarkSignature(
      name = "depset",
      returnType = SkylarkNestedSet.class,
      doc =
          "Creates a <a href=\"depset.html\">depset</a>. The <code>direct</code> parameter is a "
              + "list of direct elements of the depset, and <code>transitive</code> parameter is "
              + "a list of depsets whose elements become indirect elements of the created depset. "
              + "The order in which elements are returned when the depset is converted to a list "
              + "is specified by the <code>order</code> parameter. "
              + "See the <a href=\"../depsets.md\">Depsets overview</a> for more information. "
              + "<p> All elements (direct and indirect) of a depset must be of the same type. "
              + "<p> The order of the created depset should be <i>compatible</i> with the order of "
              + "its <code>transitive</code> depsets. <code>\"default\"</code> order is compatible "
              + "with any other order, all other orders are only compatible with themselves."
              + "<p> Note on backward/forward compatibility. This function currently accepts a "
              + "positional <code>items</code> parameter. It is deprecated and will be removed "
              + "in the future, and after its removal <code>direct</code> will become a sole "
              + "positional parameter of the <code>depset</code> function. Thus, both of the "
              + "following calls are equivalent and future-proof:<br>"
              + "<pre class=language-python>"
              + "depset(['a', 'b'], transitive = [...])\n"
              + "depset(direct = ['a', 'b'], transitive = [...])\n"
              + "</pre>",
      parameters = {
        @Param(
            name = "items",
            type = Object.class,
            defaultValue = "[]",
            doc =
                "Deprecated: Either an iterable whose items become the direct elements of "
                    + "the new depset, in left-to-right order, or else a depset that becomes "
                    + "a transitive element of the new depset. In the latter case, "
                    + "<code>transitive</code> cannot be specified."),
        @Param(
            name = "order",
            type = String.class,
            defaultValue = "\"default\"",
            doc =
                "The traversal strategy for the new depset. See "
                    + "<a href=\"depset.html\">here</a> for the possible values."),
        @Param(
            name = "direct",
            type = SkylarkList.class,
            defaultValue = "None",
            positional = false,
            named = true,
            noneable = true,
            doc = "A list of <i>direct</i> elements of a depset."),
        @Param(
            name = "transitive",
            named = true,
            positional = false,
            type = SkylarkList.class,
            generic1 = SkylarkNestedSet.class,
            noneable = true,
            doc = "A list of depsets whose elements will become indirect elements of the depset.",
            defaultValue = "None")
      },
      useLocation = true)
  private static final BuiltinFunction depset =
      new BuiltinFunction("depset") {
        public SkylarkNestedSet invoke(
            Object items, String orderString, Object direct, Object transitive, Location loc)
            throws EvalException {
          Order order;
          try {
            order = Order.parse(orderString);
          } catch (IllegalArgumentException ex) {
            throw new EvalException(loc, ex);
          }

          if (transitive == Runtime.NONE && direct == Runtime.NONE) {
            // Legacy behavior.
            return SkylarkNestedSet.of(order, items, loc);
          }

          if (direct != Runtime.NONE && !isEmptySkylarkList(items)) {
            throw new EvalException(
                loc, "Do not pass both 'direct' and 'items' argument to depset constructor.");
          }

          // Non-legacy behavior: either 'transitive' or 'direct' were specified.
          Iterable<Object> directElements;
          if (direct != Runtime.NONE) {
            directElements = ((SkylarkList<?>) direct).getContents(Object.class, "direct");
          } else {
            SkylarkType.checkType(items, SkylarkList.class, "items");
            directElements = ((SkylarkList<?>) items).getContents(Object.class, "items");
          }

          Iterable<SkylarkNestedSet> transitiveList;
          if (transitive != Runtime.NONE) {
            SkylarkType.checkType(transitive, SkylarkList.class, "transitive");
            transitiveList =
                ((SkylarkList<?>) transitive).getContents(SkylarkNestedSet.class, "transitive");
          } else {
            transitiveList = ImmutableList.of();
          }
          SkylarkNestedSet.Builder builder = SkylarkNestedSet.builder(order, loc);
          for (Object directElement : directElements) {
            builder.addDirect(directElement);
          }
          for (SkylarkNestedSet transitiveSet : transitiveList) {
            builder.addTransitive(transitiveSet);
          }
          return builder.build();
        }
      };

  private static boolean isEmptySkylarkList(Object o) {
    return o instanceof SkylarkList && ((SkylarkList) o).isEmpty();
  }

  /**
   * Returns a function-value implementing "select" (i.e. configurable attributes) in the specified
   * package context.
   */
  @SkylarkSignature(
      name = "select",
      doc =
          "<code>select()</code> is the helper function that makes a rule attribute "
              + "<a href=\"$BE_ROOT/common-definitions.html#configurable-attributes\">"
              + "configurable</a>. See "
              + "<a href=\"$BE_ROOT/functions.html#select\">build encyclopedia</a> for details.",
      parameters = {
        @Param(name = "x", type = SkylarkDict.class, doc = "The parameter to convert."),
        @Param(
            name = "no_match_error",
            type = String.class,
            defaultValue = "''",
            doc = "Optional custom error to report if no condition matches.")
      },
      useLocation = true)
  private static final BuiltinFunction select =
      new BuiltinFunction("select") {
        public Object invoke(SkylarkDict<?, ?> dict, String noMatchError, Location loc)
            throws EvalException {
          for (Object key : dict.keySet()) {
            if (!(key instanceof String)) {
              throw new EvalException(
                  loc, String.format("Invalid key: %s. select keys must be label references", key));
            }
          }
          return SelectorList.of(new SelectorValue(dict, noMatchError));
        }
      };

  @SkylarkSignature(
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
      returnType = MutableList.class,
      useLocation = true,
      useEnvironment = true)
  private static final BuiltinFunction zip =
      new BuiltinFunction("zip") {
        public MutableList<?> invoke(SkylarkList<?> args, Location loc, Environment env)
            throws EvalException {
          Iterator<?>[] iterators = new Iterator<?>[args.size()];
          for (int i = 0; i < args.size(); i++) {
            iterators[i] = EvalUtils.toIterable(args.get(i), loc, env).iterator();
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
          return MutableList.wrapUnsafe(env, result);
        }
      };

  /** Skylark int type. */
  @SkylarkModule(
      name = "int",
      category = SkylarkModuleCategory.BUILTIN,
      doc =
          "A type to represent integers. It can represent any number between -2147483648 and "
              + "2147483647 (included). "
              + "Examples of int values:<br>"
              + "<pre class=\"language-python\">"
              + "153\n"
              + "0x2A  # hexadecimal literal\n"
              + "054  # octal literal\n"
              + "23 * 2 + 5\n"
              + "100 / -7\n"
              + "100 % -7  # -5 (unlike in some other languages)\n"
              + "int(\"18\")\n"
              + "</pre>")
  public static final class IntModule {}

  /** Skylark bool type. */
  @SkylarkModule(
      name = "bool",
      category = SkylarkModuleCategory.BUILTIN,
      doc =
          "A type to represent booleans. There are only two possible values: "
              + "<a href=\"globals.html#True\">True</a> and "
              + "<a href=\"globals.html#False\">False</a>. "
              + "Any value can be converted to a boolean using the "
              + "<a href=\"globals.html#bool\">bool</a> function.")
  public static final class BoolModule {}

  /** Adds bindings for all the builtin functions of this class to the given map builder. */
  public static void addBindingsToBuilder(ImmutableMap.Builder<String, Object> builder) {
    for (BaseFunction function : allFunctions) {
      builder.put(function.getName(), function);
    }
  }

  private static final ImmutableList<BaseFunction> allFunctions =
      ImmutableList.of(
          all, any, bool, depset, dict, dir, fail, getattr, hasattr, hash, enumerate, int_, len,
          list, max, min, print, range, repr, reversed, select, sorted, str, tuple, type, zip);

  static {
    SkylarkSignatureProcessor.configureSkylarkFunctions(MethodLibrary.class);
  }
}
