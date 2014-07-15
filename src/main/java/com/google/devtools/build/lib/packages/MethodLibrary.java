// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.MixedModeFunction;
import com.google.devtools.build.lib.syntax.PositionalFunction;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A helper class containing built in functions for the Build and the Build Extension Language.
 */
public class MethodLibrary {

  private MethodLibrary() {}

  // Convert string index in the same way Python does.
  // If index is negative, starts from the end.
  // If index is outside bounds, it is restricted to the valid range.
  private static int getPythonStringIndex(int index, int stringLength) {
    if (index < 0) {
      index += stringLength;
    }
    return Math.max(Math.min(index, stringLength), 0);
  }

  // Emulate Python substring function
  // It converts out of range indices, and never fails
  private static String getPythonSubstring(String str, int start, int end) {
    start = getPythonStringIndex(start, str.length());
    end = getPythonStringIndex(end, str.length());
    if (start > end) {
      return "";
    } else {
      return str.substring(start, end);
    }
  }

  public static int getListIndex(Object key, List<?> list, FuncallExpression ast)
      throws ConversionException, EvalException {
    // Get the nth element in the list
    int index = Type.INTEGER.convert(key, "index operand");
    if (index < 0) {
      index += list.size();
    }
    if (index < 0 || index >= list.size()) {
      throw new EvalException(ast.getLocation(), "List index out of range (index is "
          + index + ", but list has " + list.size() + " elements)");
    }
    return index;
  }

    // supported string methods

  @SkylarkCallable(
      doc = "Returns a string in which the string elements of the argument have been "
          + "joined by this string as a separator.")
  private static Function join = new PositionalFunction("join", 2, 2) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(args.get(0), "'join' operand");
      List<?> seq = Type.OBJECT_LIST.convert(args.get(1), "'join' argument");
      StringBuilder sb = new StringBuilder();
      for (Iterator<?> i = seq.iterator(); i.hasNext();) {
        sb.append(i.next().toString());
        if (i.hasNext()) {
          sb.append(thiz);
        }
      }
      return sb.toString();
    }
  };

  @SkylarkCallable(
      doc = "Returns the lower case version of this string.")
  private static Function lower = new PositionalFunction("lower", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(args.get(0), "'lower' operand");
      return thiz.toLowerCase();
    }
  };

  @SkylarkCallable(
      doc = "Returns a copy of the string in which the occurrences "
          + "of <i>old</i> have been replaced with <i>new</i>, optionally restricting the number "
          + "of replacements to <i>maxsplit</i>.")
  private static Function replace =
    new MixedModeFunction("replace", ImmutableList.of("this", "old", "new", "maxsplit"), 3, false) {
    @Override
    public Object call(Object[] namedArguments, List<Object> positionalArguments,
        Map<String, Object> keywordArguments, FuncallExpression ast) throws EvalException,
        ConversionException {
      String thiz = Type.STRING.convert(namedArguments[0], "'replace' operand");
      String old = Type.STRING.convert(namedArguments[1], "'replace' argument");
      String neww = Type.STRING.convert(namedArguments[2], "'replace' argument");
      int maxsplit =
          namedArguments[3] != null ? Type.INTEGER.convert(namedArguments[3], "'replace' argument")
              : Integer.MAX_VALUE;
      StringBuffer sb = new StringBuffer();
      try {
        Matcher m = Pattern.compile(old, Pattern.LITERAL).matcher(thiz);
        for (int i = 0; i < maxsplit && m.find(); i++) {
          m.appendReplacement(sb, Matcher.quoteReplacement(neww));
        }
        m.appendTail(sb);
      } catch (IllegalStateException e) {
        throw new EvalException(ast.getLocation(), e.getMessage() + " in call to replace");
      }
      return sb.toString();
    }
  };

  @SkylarkCallable(
      doc = "Returns a list of all the words in the string, using <i>sep</i>  "
          + "as the separator, optionally limiting the number of splits to <i>maxsplit</i>.")
  private static Function split = new MixedModeFunction("split",
      ImmutableList.of("this", "sep", "maxsplit"), 1, false) {
    @Override
    public Object call(Object[] namedArguments,
        List<Object> positionalArguments,
        Map<String, Object> keywordArguments,
        FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(namedArguments[0], "'split' operand");
      String sep = namedArguments[1] != null
          ? Type.STRING.convert(namedArguments[1], "'split' argument")
          : " ";
      int maxsplit = namedArguments[2] != null
          ? Type.INTEGER.convert(namedArguments[2], "'split' argument") + 1 // last is remainder
          : -1;
      String[] ss = Pattern.compile(sep, Pattern.LITERAL).split(thiz,
                                                                maxsplit);
      return java.util.Arrays.asList(ss);
    }
  };

  @SkylarkCallable(
      doc = "Returns the last index where <i>sub</i> is found, "
          + "or -1 if no such index exists, optionally restricting to [<i>start</i>:<i>end</i>], "
          + "<i>start</i> being inclusive and <i>end</i> being exclusive.")
  private static Function rfind =
      new MixedModeFunction("rfind", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
        @Override
        public Object call(Object[] namedArguments, List<Object> positionalArguments,
            Map<String, Object> keywordArguments, FuncallExpression ast)
            throws ConversionException {
          String thiz = Type.STRING.convert(namedArguments[0], "'rfind' operand");
          String sub = Type.STRING.convert(namedArguments[1], "'rfind' argument");
          int start = 0;
          if (namedArguments[2] != null) {
            start = Type.INTEGER.convert(namedArguments[2], "'rfind' argument");
          }
          int end = thiz.length();
          if (namedArguments[3] != null) {
            end = Type.INTEGER.convert(namedArguments[3], "'rfind' argument");
          }
          int subpos = getPythonSubstring(thiz, start, end).lastIndexOf(sub);
          start = getPythonStringIndex(start, thiz.length());
          return subpos < 0 ? subpos : subpos + start;
        }
      };

  @SkylarkCallable(
      doc = "Returns the first index where <i>sub</i> is found, "
          + "or -1 if no such index exists, optionally restricting to [<i>start</i>:<i>end</i>, "
          + "<i>start</i> being inclusive and <i>end</i> being exclusive.")
  private static Function find =
      new MixedModeFunction("find", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
        @Override
        public Object call(Object[] namedArguments, List<Object> positionalArguments,
            Map<String, Object> keywordArguments, FuncallExpression ast)
            throws ConversionException {
          String thiz = Type.STRING.convert(namedArguments[0], "'find' operand");
          String sub = Type.STRING.convert(namedArguments[1], "'find' argument");
          int start = 0;
          if (namedArguments[2] != null) {
            start = Type.INTEGER.convert(namedArguments[2], "'find' argument");
          }
          int end = thiz.length();
          if (namedArguments[3] != null) {
            end = Type.INTEGER.convert(namedArguments[3], "'find' argument");
          }
          int subpos = getPythonSubstring(thiz, start, end).indexOf(sub);
          start = getPythonStringIndex(start, thiz.length());
          return subpos < 0 ? subpos : subpos + start;
        }
      };

  @SkylarkCallable(
      doc = "Returns True if the string ends with <i>sub</i>, "
          + "otherwise False, optionally restricting to [<i>start</i>:<i>end</i>], "
          + "<i>start</i> being inclusive and <i>end</i> being exclusive.")
  private static Function endswith =
      new MixedModeFunction("endswith", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
        @Override
        public Object call(Object[] namedArguments, List<Object> positionalArguments,
            Map<String, Object> keywordArguments, FuncallExpression ast)
            throws ConversionException {
          String thiz = Type.STRING.convert(namedArguments[0], "'endswith' operand");
          String sub = Type.STRING.convert(namedArguments[1], "'endswith' argument");
          int start = 0;
          if (namedArguments[2] != null) {
            start = Type.INTEGER.convert(namedArguments[2], "'endswith' argument");
          }
          int end = thiz.length();
          if (namedArguments[3] != null) {
            end = Type.INTEGER.convert(namedArguments[3], "");
          }

          return getPythonSubstring(thiz, start, end).endsWith(sub);
        }
      };

  @SkylarkCallable(
      doc = "Returns True if the string starts with <i>sub</i>, "
          + "otherwise False, optionally restricting to [<i>start</i>:<i>end</i>], "
          + "<i>start</i> being inclusive and <i>end</i> being exclusive.")
  private static Function startswith =
    new MixedModeFunction("startswith", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
    @Override
    public Object call(Object[] namedArguments, List<Object> positionalArguments,
        Map<String, Object> keywordArguments, FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(namedArguments[0], "'startswith' operand");
      String sub = Type.STRING.convert(namedArguments[1], "'startswith' argument");
      int start = 0;
      if (namedArguments[2] != null) {
        start = Type.INTEGER.convert(namedArguments[2], "'startswith' argument");
      }
      int end = thiz.length();
      if (namedArguments[3] != null) {
        end = Type.INTEGER.convert(namedArguments[3], "'startswith' argument");
      }
      return getPythonSubstring(thiz, start, end).startsWith(sub);
    }
  };

  // TODO(bazel-team): Maybe support an argument to tell the type of the whitespace.
  @SkylarkCallable(
      doc = "Returns a copy of the string in which all whitespace characters "
          + "have been stripped from the beginning and the end of the string.")
  private static Function strip =
      new MixedModeFunction("strip", ImmutableList.of("this"), 1, false) {
        @Override
        public Object call(Object[] namedArguments, List<Object> positionalArguments,
            Map<String, Object> keywordArguments, FuncallExpression ast)
            throws ConversionException {
          String operand = Type.STRING.convert(namedArguments[0], "'strip' operand");
          return operand.trim();
        }
      };

  // substring operator
  @SkylarkCallable(hidden = true,
      doc = "String[<i>start</i>:<i>end</i>] returns a substring.")
  private static Function substring = new PositionalFunction("$substring", 3, 3) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(args.get(0), "substring operand");
      int left = Type.INTEGER.convert(args.get(1), "substring operand");
      int right = Type.INTEGER.convert(args.get(2), "substring operand");
      return getPythonSubstring(thiz, left, right);
    }
  };

  // supported list methods
  @SkylarkCallable(hidden = true,
      doc = "Adds an item to the end of the list.")
  private static Function append = new PositionalFunction("append", 2, 2) {
    // @SuppressWarnings("unchecked")
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException,
        ConversionException {
      List<Object> thiz = Type.OBJECT_LIST.convert(args.get(0), "'append' operand");
      try {
        thiz.add(args.get(1));
      } catch (UnsupportedOperationException e) {
        throw new EvalException(ast.getLocation(), "cannot append to a read-only collection");
      }
      return 0;
    }
  };

  @SkylarkCallable(hidden = true,
      doc = "Adds all items to the end of the list.")
  private static Function extend = new PositionalFunction("extend", 2, 2) {
    // @SuppressWarnings("unchecked")
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException,
        ConversionException {
      List<Object> thiz = Type.OBJECT_LIST.convert(args.get(0), "'extend' operand");
      List<Object> l = Type.OBJECT_LIST.convert(args.get(1), "'extend' argument");
      try {
        thiz.addAll(l);
      } catch (UnsupportedOperationException e) {
        throw new EvalException(ast.getLocation(), "cannot extend a read-only collection");
      }
      return 0;
    }
  };

  // dictionary access operator
  @SkylarkBuiltin(name = "$index", hidden = true,
      doc = "Returns the nth element of a list or string, "
          + "or looks up a value in a dictionary.")
  private static Function index = new PositionalFunction("$index", 2, 2) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException,
        ConversionException {
      Object collectionCandidate = args.get(0);
      Object key = args.get(1);

      if (collectionCandidate instanceof Map<?, ?>) {
        Map<?, ?> dictionary = (Map<?, ?>) collectionCandidate;
        if (!dictionary.containsKey(key)) {
          throw new EvalException(ast.getLocation(), "Key '" + key + "' not found in dictionary");
        }
        return dictionary.get(key);
      } else if (collectionCandidate instanceof List<?>) {

        List<Object> list = Type.OBJECT_LIST.convert(collectionCandidate, "index operand");

        if (!list.isEmpty()) {
          int index = getListIndex(key, list, ast);
          return list.get(index);
        }

        throw new EvalException(ast.getLocation(), "List is empty");
      } else {
        throw new EvalException(ast.getLocation(), String.format(
            "Unsupported datatype (%s) for indexing, only works for dict and list",
            EvalUtils.getDatatypeName(collectionCandidate)));
      }
    }
  };

  // unary minus
  @SkylarkBuiltin(name = "-", hidden = true, doc = "Unary minus operator.")
  private static Function minus = new PositionalFunction("-", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
      int num = Type.INTEGER.convert(args.get(0), "'unary minus' argument");
      return -num;
    }
  };

  // TODO(bazel-team): support sets properly
  // creates an empty set
  @SkylarkBuiltin(name = "set", doc = "Creates an empty set.")
  private static Function set = new PositionalFunction("set", 0, 0) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
      return new HashSet<Object>();
    }
  };

  @SkylarkBuiltin(name = "len", doc = "Returns the length of a string, list, tuple or dictionary.")
  private static Function len = new PositionalFunction("len", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException {
      Object arg = args.get(0);
      int l = EvalUtils.size(arg);
      if (l == -1) {
        throw new EvalException(ast.getLocation(),
            EvalUtils.getDatatypeName(arg) + " is not iterable");
      }
      return l;
    }
  };

  @SkylarkBuiltin(name = "str", doc = "Converts an object to string")
  private static Function str = new PositionalFunction("str", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException {
      return EvalUtils.printValue(args.get(0));
    }
  };

  @SkylarkBuiltin(name = "bool", doc = "Converts an object to boolean")
  private static Function bool = new PositionalFunction("bool", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException {
      return EvalUtils.toBoolean(args.get(0));
    }
  };

  @SkylarkBuiltin(name = "struct", doc = "Creates a struct using the keyword arguments as fields.")
  private static Function struct = new AbstractFunction("struct") {

    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      if (args.size() > 0) {
        throw new EvalException(ast.getLocation(), "struct only supports keyword arguments");
      }
      return new ClassObject(kwargs);
    }
  };

  @SkylarkBuiltin(name = "nset",
      doc = "Creates a nested set from the <i>items</i>. "
          + "The nesting is applied to other nested sets among <i>items</i>.")
  private static final Function nset = new PositionalFunction("nset", 1, 2) {

    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException,
        ConversionException {
      // TODO(bazel-team): Investigate if enums or constants can be used here in the argument.
      String orderString = Type.STRING.cast(args.get(0));
      Order order;
      try {
        order = Order.valueOf(orderString);
      } catch (IllegalArgumentException e) {
        throw new EvalException(ast.getLocation(), "Invalid order " + orderString);
      }
      if (args.size() == 2) {
        return new SkylarkNestedSet(order, args.get(1), ast.getLocation());
      } else {
        return new SkylarkNestedSet(order, ImmutableList.of(), ast.getLocation());
      }
    }
  };

  @SkylarkBuiltin(name = "String", doc = "")
  public static final Map<Function, Class<?>> stringFunctions = ImmutableMap
      .<Function, Class<?>>builder()
      .put(join, String.class)
      .put(lower, String.class)
      .put(replace, String.class)
      .put(split, List.class)
      .put(rfind, Integer.class)
      .put(find, Integer.class)
      .put(endswith, Boolean.class)
      .put(startswith, Boolean.class)
      .put(strip, String.class)
      .put(substring, String.class)
      .build();

  @SkylarkBuiltin(name = "List", doc = "")
  public static final List<Function> listFunctions = ImmutableList
      .<Function>builder()
      .add(append)
      .add(extend)
      .build();

  private static final Map<Function, Class<?>> pureFunctions = ImmutableMap
      .<Function, Class<?>>builder()
      .putAll(stringFunctions)
      .put(index, Object.class)
      .put(minus, Integer.class)
      .put(set, Set.class)
      .put(len, Integer.class)
      .put(str, String.class)
      .put(bool, Boolean.class)
      .build();

  private static final Map<Function, Class<?>> skylarkFunctions = ImmutableMap
      .<Function, Class<?>>builder()
      .putAll(pureFunctions)
      .put(struct, ClassObject.class)
      .put(nset, SkylarkNestedSet.class)
      .build();

  // TODO(bazel-team): listFunctions are not allowed in Skylark extensions (use += instead).
  // It is allowed in BUILD files only for backward-compatibility.
  private static final List<Function> functions = ImmutableList
      .<Function>builder()
      .addAll(pureFunctions.keySet())
      .addAll(listFunctions)
      .build();

  /**
   * Set up a given environment for supported class methods.
   */
  public static void setupMethodEnvironment(Environment env) {
    for (Function function : env.isSkylarkEnabled() ? skylarkFunctions.keySet() : functions) {
      env.update(function.getName(), function);
    }
  }

  public static void setupValidationEnvironment(ImmutableMap.Builder<String, Class<?>> builder) {
    for (Map.Entry<Function, Class<?>> function : skylarkFunctions.entrySet()) {
      String name = function.getKey().getName();
      builder.put(name, Function.class);
      builder.put(name + ".return", function.getValue());
    }
  }
}
