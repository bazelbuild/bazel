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

import static com.google.devtools.build.lib.syntax.SkylarkFunction.cast;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.AbstractFunction.NoArgFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.DotExpression;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.MixedModeFunction;
import com.google.devtools.build.lib.syntax.SelectorList;
import com.google.devtools.build.lib.syntax.SelectorValue;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin.Param;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.ExecutionException;
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
  private static int getClampedIndex(int index, int length) {
    if (index < 0) {
      index += length;
    }
    return Math.max(Math.min(index, length), 0);
  }

  // Emulate Python substring function
  // It converts out of range indices, and never fails
  private static String getPythonSubstring(String str, int start, int end) {
    start = getClampedIndex(start, str.length());
    end = getClampedIndex(end, str.length());
    if (start > end) {
      return "";
    } else {
      return str.substring(start, end);
    }
  }

  public static int getListIndex(Object key, int listSize, FuncallExpression ast)
      throws ConversionException, EvalException {
    // Get the nth element in the list
    int index = Type.INTEGER.convert(key, "index operand");
    if (index < 0) {
      index += listSize;
    }
    if (index < 0 || index >= listSize) {
      throw new EvalException(ast.getLocation(), "List index out of range (index is "
          + index + ", but list has " + listSize + " elements)");
    }
    return index;
  }

    // supported string methods

  @SkylarkBuiltin(name = "join", objectType = StringModule.class, returnType = String.class,
      doc = "Returns a string in which the string elements of the argument have been "
          + "joined by this string as a separator. Example:<br>"
          + "<pre class=\"language-python\">\"|\".join([\"a\", \"b\", \"c\"]) == \"a|b|c\"</pre>",
      mandatoryParams = {
      @Param(name = "elements", type = SkylarkList.class, doc = "The objects to join.")})
  private static Function join = new MixedModeFunction("join",
      ImmutableList.of("this", "elements"), 2, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws ConversionException {
      String thisString = Type.STRING.convert(args[0], "'join' operand");
      List<?> seq = Type.OBJECT_LIST.convert(args[1], "'join' argument");
      return Joiner.on(thisString).join(seq);
    }};

  @SkylarkBuiltin(name = "lower", objectType = StringModule.class, returnType = String.class,
      doc = "Returns the lower case version of this string.")
      private static Function lower = new MixedModeFunction("lower",
          ImmutableList.of("this"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(args[0], "'lower' operand");
      return thiz.toLowerCase();
    }
  };

  @SkylarkBuiltin(name = "upper", objectType = StringModule.class, returnType = String.class,
      doc = "Returns the upper case version of this string.")
    private static Function upper = new MixedModeFunction("upper",
        ImmutableList.of("this"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(args[0], "'upper' operand");
      return thiz.toUpperCase();
    }
  };

  @SkylarkBuiltin(name = "replace", objectType = StringModule.class, returnType = String.class,
      doc = "Returns a copy of the string in which the occurrences "
          + "of <code>old</code> have been replaced with <code>new</code>, optionally restricting "
          + "the number of replacements to <code>maxsplit</code>.",
      mandatoryParams = {
      @Param(name = "old", type = String.class, doc = "The string to be replaced."),
      @Param(name = "new", type = String.class, doc = "The string to replace with.")},
      optionalParams = {
      @Param(name = "maxsplit", type = Integer.class, doc = "The maximum number of replacements.")})
  private static Function replace =
    new MixedModeFunction("replace", ImmutableList.of("this", "old", "new", "maxsplit"), 3, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException,
        ConversionException {
      String thiz = Type.STRING.convert(args[0], "'replace' operand");
      String old = Type.STRING.convert(args[1], "'replace' argument");
      String neww = Type.STRING.convert(args[2], "'replace' argument");
      int maxsplit =
          args[3] != null ? Type.INTEGER.convert(args[3], "'replace' argument")
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

  @SkylarkBuiltin(name = "split", objectType = StringModule.class, returnType = SkylarkList.class,
      doc = "Returns a list of all the words in the string, using <code>sep</code>  "
          + "as the separator, optionally limiting the number of splits to <code>maxsplit</code>.",
      optionalParams = {
      @Param(name = "sep", type = String.class,
          doc = "The string to split on, default is space (\" \")."),
      @Param(name = "maxsplit", type = Integer.class, doc = "The maximum number of splits.")})
  private static Function split = new MixedModeFunction("split",
      ImmutableList.of("this", "sep", "maxsplit"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast, Environment env)
        throws ConversionException {
      String thiz = Type.STRING.convert(args[0], "'split' operand");
      String sep = args[1] != null
          ? Type.STRING.convert(args[1], "'split' argument")
          : " ";
      int maxsplit = args[2] != null
          ? Type.INTEGER.convert(args[2], "'split' argument") + 1 // last is remainder
          : -1;
      String[] ss = Pattern.compile(sep, Pattern.LITERAL).split(thiz,
                                                                maxsplit);
      List<String> result = java.util.Arrays.asList(ss);
      return env.isSkylarkEnabled() ? SkylarkList.list(result, String.class) : result;
    }
  };

  // Common implementation for find, rfind, index, rindex.
  // forward is true iff we want to return the last matching index.
  private static int stringFind(String functionName, boolean forward, Object[] args)
      throws ConversionException {
    String thiz = Type.STRING.convert(args[0], functionName + " operand");
    String sub = Type.STRING.convert(args[1], functionName + " argument");
    int start = 0;
    if (args[2] != null) {
      start = Type.INTEGER.convert(args[2], functionName + " argument");
    }
    int end = thiz.length();
    if (args[3] != null) {
      end = Type.INTEGER.convert(args[3], functionName + " argument");
    }
    String substr = getPythonSubstring(thiz, start, end);
    int subpos = forward ? substr.indexOf(sub) : substr.lastIndexOf(sub);
    start = getClampedIndex(start, thiz.length());
    return subpos < 0 ? subpos : subpos + start;
  }

  @SkylarkBuiltin(name = "rfind", objectType = StringModule.class, returnType = Integer.class,
      doc = "Returns the last index where <code>sub</code> is found, "
          + "or -1 if no such index exists, optionally restricting to "
          + "[<code>start</code>:<code>end</code>], "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      mandatoryParams = {
      @Param(name = "sub", type = String.class, doc = "The substring to find.")},
      optionalParams = {
      @Param(name = "start", type = Integer.class, doc = "Restrict to search from this position."),
      @Param(name = "end", type = Integer.class, doc = "Restrict to search before this position.")})
  private static Function rfind =
      new MixedModeFunction("rfind", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast) throws ConversionException {
          return stringFind("rfind", false, args);
        }
      };

  @SkylarkBuiltin(name = "find", objectType = StringModule.class, returnType = Integer.class,
      doc = "Returns the first index where <code>sub</code> is found, "
          + "or -1 if no such index exists, optionally restricting to "
          + "[<code>start</code>:<code>end]</code>, "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      mandatoryParams = {
      @Param(name = "sub", type = String.class, doc = "The substring to find.")},
      optionalParams = {
      @Param(name = "start", type = Integer.class, doc = "Restrict to search from this position."),
      @Param(name = "end", type = Integer.class, doc = "Restrict to search before this position.")})
  private static Function find =
      new MixedModeFunction("find", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast)
            throws ConversionException {
          return stringFind("find", true, args);
        }
      };

  @SkylarkBuiltin(name = "rindex", objectType = StringModule.class, returnType = Integer.class,
      doc = "Returns the last index where <code>sub</code> is found, "
          + "or throw an error if no such index exists, optionally restricting to "
          + "[<code>start</code>:<code>end</code>], "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      mandatoryParams = {
      @Param(name = "sub", type = String.class, doc = "The substring to find.")},
      optionalParams = {
      @Param(name = "start", type = Integer.class, doc = "Restrict to search from this position."),
      @Param(name = "end", type = Integer.class, doc = "Restrict to search before this position.")})
  private static Function rindex =
      new MixedModeFunction("rindex", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast)
            throws EvalException, ConversionException {
          int res = stringFind("rindex", false, args);
          if (res < 0) {
            throw new EvalException(ast.getLocation(),
                "substring '" + args[1] + "' not found in '" + args[0] + "'");
          }
          return res;
        }
      };

  @SkylarkBuiltin(name = "index", objectType = StringModule.class, returnType = Integer.class,
      doc = "Returns the first index where <code>sub</code> is found, "
          + "or throw an error if no such index exists, optionally restricting to "
          + "[<code>start</code>:<code>end]</code>, "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      mandatoryParams = {
      @Param(name = "sub", type = String.class, doc = "The substring to find.")},
      optionalParams = {
      @Param(name = "start", type = Integer.class, doc = "Restrict to search from this position."),
      @Param(name = "end", type = Integer.class, doc = "Restrict to search before this position.")})
  private static Function index =
      new MixedModeFunction("index", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast)
            throws EvalException, ConversionException {
          int res = stringFind("index", true, args);
          if (res < 0) {
            throw new EvalException(ast.getLocation(),
                "substring '" + args[1] + "' not found in '" + args[0] + "'");
          }
          return res;
        }
      };

  @SkylarkBuiltin(name = "count", objectType = StringModule.class, returnType = Integer.class,
      doc = "Returns the number of (non-overlapping) occurrences of substring <code>sub</code> in "
          + "string, optionally restricting to [<code>start</code>:<code>end</code>], "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      mandatoryParams = {
      @Param(name = "sub", type = String.class, doc = "The substring to count.")},
      optionalParams = {
      @Param(name = "start", type = Integer.class, doc = "Restrict to search from this position."),
      @Param(name = "end", type = Integer.class, doc = "Restrict to search before this position.")})
  private static Function count =
      new MixedModeFunction("count", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast)
            throws ConversionException {
          String thiz = Type.STRING.convert(args[0], "'count' operand");
          String sub = Type.STRING.convert(args[1], "'count' argument");
          int start = 0;
          if (args[2] != null) {
            start = Type.INTEGER.convert(args[2], "'count' argument");
          }
          int end = thiz.length();
          if (args[3] != null) {
            end = Type.INTEGER.convert(args[3], "'count' argument");
          }
          String str = getPythonSubstring(thiz, start, end);
          if (sub.isEmpty()) {
            return str.length() + 1;
          }
          int count = 0;
          int index = -1;
          while ((index = str.indexOf(sub)) >= 0) {
            count++;
            str = str.substring(index + sub.length());
          }
          return count;
        }
      };

  @SkylarkBuiltin(name = "endswith", objectType = StringModule.class, returnType = Boolean.class,
      doc = "Returns True if the string ends with <code>sub</code>, "
          + "otherwise False, optionally restricting to [<code>start</code>:<code>end</code>], "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      mandatoryParams = {
      @Param(name = "sub", type = String.class, doc = "The substring to check.")},
      optionalParams = {
      @Param(name = "start", type = Integer.class, doc = "Test beginning at this position."),
      @Param(name = "end", type = Integer.class, doc = "Stop comparing at this position.")})
  private static Function endswith =
      new MixedModeFunction("endswith", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast)
            throws ConversionException {
          String thiz = Type.STRING.convert(args[0], "'endswith' operand");
          String sub = Type.STRING.convert(args[1], "'endswith' argument");
          int start = 0;
          if (args[2] != null) {
            start = Type.INTEGER.convert(args[2], "'endswith' argument");
          }
          int end = thiz.length();
          if (args[3] != null) {
            end = Type.INTEGER.convert(args[3], "");
          }

          return getPythonSubstring(thiz, start, end).endsWith(sub);
        }
      };

  @SkylarkBuiltin(name = "startswith", objectType = StringModule.class, returnType = Boolean.class,
      doc = "Returns True if the string starts with <code>sub</code>, "
          + "otherwise False, optionally restricting to [<code>start</code>:<code>end</code>], "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      mandatoryParams = {
      @Param(name = "sub", type = String.class, doc = "The substring to check.")},
      optionalParams = {
      @Param(name = "start", type = Integer.class, doc = "Test beginning at this position."),
      @Param(name = "end", type = Integer.class, doc = "Stop comparing at this position.")})
  private static Function startswith =
    new MixedModeFunction("startswith", ImmutableList.of("this", "sub", "start", "end"), 2, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(args[0], "'startswith' operand");
      String sub = Type.STRING.convert(args[1], "'startswith' argument");
      int start = 0;
      if (args[2] != null) {
        start = Type.INTEGER.convert(args[2], "'startswith' argument");
      }
      int end = thiz.length();
      if (args[3] != null) {
        end = Type.INTEGER.convert(args[3], "'startswith' argument");
      }
      return getPythonSubstring(thiz, start, end).startsWith(sub);
    }
  };

  // TODO(bazel-team): Maybe support an argument to tell the type of the whitespace.
  @SkylarkBuiltin(name = "strip", objectType = StringModule.class, returnType = String.class,
      doc = "Returns a copy of the string in which all whitespace characters "
          + "have been stripped from the beginning and the end of the string.")
  private static Function strip =
      new MixedModeFunction("strip", ImmutableList.of("this"), 1, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast)
            throws ConversionException {
          String operand = Type.STRING.convert(args[0], "'strip' operand");
          return operand.trim();
        }
      };

  // slice operator
  @SkylarkBuiltin(name = "$slice", documented = false,
      doc = "x[<code>start</code>:<code>end</code>] returns a slice or a list slice.")
  private static Function slice = new MixedModeFunction("$slice",
      ImmutableList.of("this", "start", "end"), 3, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast, Environment env)
        throws EvalException, ConversionException {
      int left = Type.INTEGER.convert(args[1], "start operand");
      int right = Type.INTEGER.convert(args[2], "end operand");

      // Substring
      if (args[0] instanceof String) {
        String thiz = Type.STRING.convert(args[0], "substring operand");
        return getPythonSubstring(thiz, left, right);
      }

      // List slice
      List<Object> list = Type.OBJECT_LIST.convert(args[0], "list operand");
      left = getClampedIndex(left, list.size());
      right = getClampedIndex(right, list.size());

      List<Object> result = Lists.newArrayList();
      for (int i = left; i < right; i++) {
        result.add(list.get(i));
      }
      return convert(result, env, ast.getLocation());
    }
  };

  // supported list methods
  @SkylarkBuiltin(name = "sorted", doc = "Sort a collection.")
  private static Function sorted = new MixedModeFunction("sorted",
      ImmutableList.of("this"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast, Environment env)
        throws EvalException, ConversionException {
      List<Object> thiz = Type.OBJECT_LIST.convert(args[0], "'sorted' operand");
      try {
        thiz = Ordering.from(EvalUtils.SKYLARK_COMPARATOR).sortedCopy(thiz);
      } catch (EvalUtils.ComparisonException e) {
        throw new EvalException(ast.getLocation(), e);
      }
      return convert(thiz, env, ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "append", documented = false,
      doc = "Adds an item to the end of the list.")
  private static Function append = new MixedModeFunction("append",
      ImmutableList.of("this", "x"), 2, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException,
        ConversionException {
      List<Object> thiz = Type.OBJECT_LIST.convert(args[0], "'append' operand");
      thiz.add(args[1]);
      return Environment.NONE;
    }
  };

  @SkylarkBuiltin(name = "extend", documented = false,
      doc = "Adds all items to the end of the list.")
  private static Function extend = new MixedModeFunction("extend",
          ImmutableList.of("this", "x"), 2, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException,
        ConversionException {
      List<Object> thiz = Type.OBJECT_LIST.convert(args[0], "'extend' operand");
      List<Object> l = Type.OBJECT_LIST.convert(args[1], "'extend' argument");
      thiz.addAll(l);
      return Environment.NONE;
    }
  };

  // dictionary access operator
  @SkylarkBuiltin(name = "$index", documented = false,
      doc = "Returns the nth element of a list or string, "
          + "or looks up a value in a dictionary.")
  private static Function indexOperator = new MixedModeFunction("$index",
      ImmutableList.of("this", "index"), 2, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException,
        ConversionException {
      Object collectionCandidate = args[0];
      Object key = args[1];

      if (collectionCandidate instanceof Map<?, ?>) {
        Map<?, ?> dictionary = (Map<?, ?>) collectionCandidate;
        if (!dictionary.containsKey(key)) {
          throw new EvalException(ast.getLocation(), "Key '" + key + "' not found in dictionary");
        }
        return dictionary.get(key);
      } else if (collectionCandidate instanceof List<?>) {

        List<Object> list = Type.OBJECT_LIST.convert(collectionCandidate, "index operand");

        if (!list.isEmpty()) {
          int index = getListIndex(key, list.size(), ast);
          return list.get(index);
        }

        throw new EvalException(ast.getLocation(), "List is empty");
      } else if (collectionCandidate instanceof SkylarkList) {
        SkylarkList list = (SkylarkList) collectionCandidate;

        if (!list.isEmpty()) {
          int index = getListIndex(key, list.size(), ast);
          return list.get(index);
        }

        throw new EvalException(ast.getLocation(), "List is empty");
      } else if (collectionCandidate instanceof String) {
        String str = (String) collectionCandidate;
        int index = getListIndex(key, str.length(), ast);
        return str.substring(index, index + 1);

      } else {
        // TODO(bazel-team): This is dead code, get rid of it.
        throw new EvalException(ast.getLocation(), String.format(
            "Unsupported datatype (%s) for indexing, only works for dict and list",
            EvalUtils.getDataTypeName(collectionCandidate)));
      }
    }
  };

  @SkylarkBuiltin(name = "values", objectType = DictModule.class, returnType = SkylarkList.class,
      doc = "Return the list of values. Dictionaries are always sorted by their keys:"
          + "<pre class=\"language-python\">"
          + "{2: \"a\", 4: \"b\", 1: \"c\"}.values() == [\"c\", \"a\", \"b\"]</pre>\n")
  private static Function values = new NoArgFunction("values") {
    @Override
    public Object call(Object self, FuncallExpression ast, Environment env)
        throws EvalException, InterruptedException {
      // Use a TreeMap to ensure consistent ordering.
      Map<?, ?> dict = new TreeMap<>((Map<?, ?>) self);
      return convert(dict.values(), env, ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "items", objectType = DictModule.class, returnType = SkylarkList.class,
      doc = "Return the list of key-value tuples. Dictionaries are always sorted by their keys:"
          + "<pre class=\"language-python\">"
          + "{2: \"a\", 4: \"b\", 1: \"c\"}.items() == [(1, \"c\"), (2, \"a\"), (4, \"b\")]"
          + "</pre>\n")
  private static Function items = new NoArgFunction("items") {
    @Override
    public Object call(Object self, FuncallExpression ast, Environment env)
        throws EvalException, InterruptedException {
      // Use a TreeMap to ensure consistent ordering.
      Map<?, ?> dict = new TreeMap<>((Map<?, ?>) self);
      List<Object> list = Lists.newArrayListWithCapacity(dict.size());
      for (Map.Entry<?, ?> entries : dict.entrySet()) {
        List<?> item = ImmutableList.of(entries.getKey(), entries.getValue());
        list.add(env.isSkylarkEnabled() ? SkylarkList.tuple(item) : item);
      }
      return convert(list, env, ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "keys", objectType = DictModule.class, returnType = SkylarkList.class,
      doc = "Return the list of keys. Dictionaries are always sorted by their keys:"
          + "<pre class=\"language-python\">{2: \"a\", 4: \"b\", 1: \"c\"}.keys() == [1, 2, 4]"
          + "</pre>\n")
  private static Function keys = new NoArgFunction("keys") {
    @Override
    public Object call(Object self, FuncallExpression ast, Environment env)
        throws EvalException, InterruptedException {
      Map<Comparable<?>, Object> dict = (Map<Comparable<?>, Object>) self;
      return convert(Ordering.natural().sortedCopy(dict.keySet()), env, ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "get", objectType = DictModule.class,
      doc = "Return the value for <code>key</code> if <code>key</code> is in the dictionary, "
          + "else <code>default</code>. If <code>default</code> is not given, it defaults to "
          + "<code>None</code>, so that this method never throws an error.",
      mandatoryParams = {
        @Param(name = "key", doc = "The key to look for.")},
      optionalParams = {
        @Param(name = "default",
            doc = "The default value to use (instead of None) if the key is not found.")})
  private static Function get =
      new MixedModeFunction("get", ImmutableList.of("this", "key", "default"), 2, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws ConversionException {
      Map<?, ?> dict = (Map<?, ?>) args[0];
      Object key = args[1];
      if (dict.containsKey(key)) {
        return dict.get(key);
      }
      if (args[2] == null) {
        return Environment.NONE;
      }
      return args[2];
    }
  };

  // TODO(bazel-team): Use the same type for both Skylark and BUILD files.
  @SuppressWarnings("unchecked")
  private static Iterable<Object> convert(Collection<?> list, Environment env, Location loc)
      throws EvalException {
    if (env.isSkylarkEnabled()) {
      return SkylarkList.list(list, loc);
    } else {
      return Lists.newArrayList(list);
    }
  }

  // unary minus
  @SkylarkBuiltin(name = "-", documented = false, doc = "Unary minus operator.")
  private static Function minus = new MixedModeFunction("-", ImmutableList.of("this"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws ConversionException {
      int num = Type.INTEGER.convert(args[0], "'unary minus' argument");
      return -num;
    }
  };

  @SkylarkBuiltin(name = "list", returnType = SkylarkList.class,
      doc = "Converts a collection (e.g. set or dictionary) to a list."
        + "<pre class=\"language-python\">list([1, 2]) == [1, 2]\n"
        + "list(set([2, 3, 2])) == [2, 3]\n"
        + "list({5: \"a\", 2: \"b\", 4: \"c\"}) == [2, 4, 5]</pre>",
      mandatoryParams = {@Param(name = "x", doc = "The object to convert.")})
    private static Function list = new MixedModeFunction("list",
        ImmutableList.of("list"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException {
      Location loc = ast.getLocation();
      return SkylarkList.list(EvalUtils.toCollection(args[0], loc), loc);
    }
  };

  @SkylarkBuiltin(name = "len", returnType = Integer.class, doc =
      "Returns the length of a string, list, tuple, set, or dictionary.",
      mandatoryParams = {@Param(name = "x", doc = "The object to check length of.")})
  private static Function len = new MixedModeFunction("len",
        ImmutableList.of("list"), 1, false) {

    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException {
      Object arg = args[0];
      int l = EvalUtils.size(arg);
      if (l == -1) {
        throw new EvalException(ast.getLocation(),
            EvalUtils.getDataTypeName(arg) + " is not iterable");
      }
      return l;
    }
  };

  @SkylarkBuiltin(name = "str", returnType = String.class, doc =
      "Converts any object to string. This is useful for debugging.",
      mandatoryParams = {@Param(name = "x", doc = "The object to convert.")})
    private static Function str = new MixedModeFunction("str", ImmutableList.of("this"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException {
      return EvalUtils.printValue(args[0]);
    }
  };

  @SkylarkBuiltin(name = "bool", returnType = Boolean.class, doc = "Converts an object to boolean. "
      + "It returns False if the object is None, False, an empty string, the number 0, or an "
      + "empty collection. Otherwise, it returns True. Similarly to Python <code>bool</code> "
      + "is also a type.",
      mandatoryParams = {@Param(name = "x", doc = "The variable to convert.")})
      private static Function bool = new MixedModeFunction("bool",
          ImmutableList.of("this"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException {
      return EvalUtils.toBoolean(args[0]);
    }
  };

  @SkylarkBuiltin(name = "int", returnType = Integer.class, doc = "Converts a string to int, "
      + "using base 10. It raises an error if the conversion fails."
      + "<pre class=\"language-python\">int(\"123\") == 123</pre>",
      mandatoryParams = {@Param(name = "x", type = String.class, doc = "The string to convert.")})
  private static Function int_ =
      new MixedModeFunction("int", ImmutableList.of("x"), 1, false) {
        @Override
        public Object call(Object[] args, FuncallExpression ast)
            throws EvalException, ConversionException {
          String str = Type.STRING.convert(args[0], "'int' operand");
          try {
            return Integer.parseInt(str);
          } catch (NumberFormatException e) {
            throw new EvalException(ast.getLocation(), "invalid literal for int(): " + str);
          }
        }
      };

  @SkylarkBuiltin(name = "struct", returnType = SkylarkClassObject.class, doc =
      "Creates an immutable struct using the keyword arguments as fields. It is used to group "
      + "multiple values together.Example:<br>"
      + "<pre class=\"language-python\">s = struct(x = 2, y = 3)\n"
      + "return s.x + s.y  # returns 5</pre>")
  private static Function struct = new AbstractFunction("struct") {

    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      if (!args.isEmpty()) {
        throw new EvalException(ast.getLocation(), "struct only supports keyword arguments");
      }
      return new SkylarkClassObject(kwargs, ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "set", returnType = SkylarkNestedSet.class,
      doc = "Creates a <a href=\"#modules.set\">set</a> from the <code>items</code>."
      + " The set supports nesting other sets of the same element"
      + " type in it. A desired iteration order can also be specified.<br>"
      + " Examples:<br><pre class=\"language-python\">set([\"a\", \"b\"])\n"
      + "set([1, 2, 3], order=\"compile\")</pre>",
      optionalParams = {
      @Param(name = "items", type = SkylarkList.class,
          doc = "The items to initialize the set with. May contain both standalone items and other"
          + " sets."),
      @Param(name = "order", type = String.class,
          doc = "The ordering strategy for the set if it's nested, "
              + "possible values are: <code>stable</code> (default), <code>compile</code>, "
              + "<code>link</code> or <code>naive_link</code>.")})
  private static final Function set =
    new MixedModeFunction("set", ImmutableList.of("items", "order"), 0, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException,
        ConversionException {
      Order order = SkylarkNestedSet.parseOrder((String) args[1], ast.getLocation());
      if (args[0] == null) {
        return new SkylarkNestedSet(order, SkylarkList.EMPTY_LIST, ast.getLocation());
      }
      return new SkylarkNestedSet(order, args[0], ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "enumerate",  returnType = SkylarkList.class,
      doc = "Return a list of pairs (two-element lists), with the index (int) and the item from"
          + " the input list.\n<pre class=\"language-python\">"
          + "enumerate([24, 21, 84]) == [[0, 24], [1, 21], [2, 84]]</pre>\n",
      mandatoryParams = {
      @Param(name = "list", type = SkylarkList.class,
          doc = "input list"),
      })
  private static Function enumerate = new MixedModeFunction("enumerate",
      ImmutableList.of("list"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException,
        ConversionException {
      // Note that enumerate is only available in Skylark.
      SkylarkList input = cast(
          args[0], SkylarkList.class, "enumerate operand", ast.getLocation());
      List<SkylarkList> result = Lists.newArrayList();
      int count = 0;
      for (Object obj : input) {
        result.add(SkylarkList.tuple(Lists.newArrayList(count, obj)));
        count++;
      }
      return SkylarkList.list(result, ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "range", returnType = SkylarkList.class,
      doc = "Creates a list where items go from <code>start</code> to <code>stop</code>, using a "
          + "<code>step</code> increment. If a single argument is provided, items will "
          + "range from 0 to that element."
          + "<pre class=\"language-python\">range(4) == [0, 1, 2, 3]\n"
          + "range(3, 9, 2) == [3, 5, 7]\n"
          + "range(3, 0, -1) == [3, 2, 1]</pre>",
      mandatoryParams = {
      @Param(name = "start", type = Integer.class,
          doc = "Value of the first element"),
      },
      optionalParams = {
      @Param(name = "stop", type = Integer.class,
          doc = "The first item <i>not</i> to be included in the resulting list; "
          + "generation of the list stops before <code>stop</code> is reached."),
      @Param(name = "step", type = Integer.class,
          doc = "The increment (default is 1). It may be negative.")})
  private static final Function range =
    new MixedModeFunction("range", ImmutableList.of("start", "stop", "step"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException,
        ConversionException {
      int start;
      int stop;
      if (args[1] == null) {
        start = 0;
        stop = Type.INTEGER.convert(args[0], "stop");
      } else {
        start = Type.INTEGER.convert(args[0], "start");
        stop = Type.INTEGER.convert(args[1], "stop");
      }
      int step = args[2] == null ? 1 : Type.INTEGER.convert(args[2], "step");
      if (step == 0) {
        throw new EvalException(ast.getLocation(), "step cannot be 0");
      }
      List<Integer> result = Lists.newArrayList();
      if (step > 0) {
        while (start < stop) {
          result.add(start);
          start += step;
        }
      } else {
        while (start > stop) {
          result.add(start);
          start += step;
        }
      }
      return SkylarkList.list(result, Integer.class);
    }
  };

  /**
   * Returns a function-value implementing "select" (i.e. configurable attributes)
   * in the specified package context.
   */
  @SkylarkBuiltin(name = "select",
      doc = "Creates a SelectorValue from the dict parameter.",
      mandatoryParams = {@Param(name = "x", type = Map.class, doc = "The parameter to convert.")})
  private static final Function select = new MixedModeFunction("select",
      ImmutableList.of("x"), 1, false) {
      @Override
      public Object call(Object[] args, FuncallExpression ast)
          throws EvalException, ConversionException {
        Object dict = args[0];
        if (!(dict instanceof Map<?, ?>)) {
          throw new EvalException(ast.getLocation(),
              "select({...}) argument isn't a dictionary");
        }
        return SelectorList.of(new SelectorValue((Map<?, ?>) dict));
      }
    };

  /**
   * Returns true if the object has a field of the given name, otherwise false.
   */
  @SkylarkBuiltin(name = "hasattr", returnType = Boolean.class,
      doc = "Returns True if the object <code>x</code> has a field of the given <code>name</code>, "
          + "otherwise False. Example:<br>"
          + "<pre class=\"language-python\">hasattr(ctx.attr, \"myattr\")</pre>",
      mandatoryParams = {
      @Param(name = "object", doc = "The object to check."),
      @Param(name = "name", type = String.class, doc = "The name of the field.")})
  private static final Function hasattr =
      new MixedModeFunction("hasattr", ImmutableList.of("object", "name"), 2, false) {

    @Override
    public Object call(Object[] args, FuncallExpression ast, Environment env)
        throws EvalException, ConversionException {
      Object obj = args[0];
      String name = cast(args[1], String.class, "name", ast.getLocation());

      if (obj instanceof ClassObject && ((ClassObject) obj).getValue(name) != null) {
        return true;
      }

      if (env.getFunctionNames(obj.getClass()).contains(name)) {
        return true;
      }

      try {
        return FuncallExpression.getMethodNames(obj.getClass()).contains(name);
      } catch (ExecutionException e) {
        // This shouldn't happen
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
    }
  };

  @SkylarkBuiltin(name = "getattr",
      doc = "Returns the struct's field of the given name if exists, otherwise <code>default</code>"
          + " if specified, otherwise rasies an error. For example, <code>getattr(x, \"foobar\")"
          + "</code> is equivalent to <code>x.foobar</code>."
          + "Example:<br>"
          + "<pre class=\"language-python\">getattr(ctx.attr, \"myattr\")\n"
          + "getattr(ctx.attr, \"myattr\", \"mydefault\")</pre>",
     mandatoryParams = {
     @Param(name = "object", doc = "The struct which's field is accessed."),
     @Param(name = "name", doc = "The name of the struct field.")},
     optionalParams = {
     @Param(name = "default", doc = "The default value to return in case the struct "
                                  + "doesn't have a field of the given name.")})
  private static final Function getattr = new MixedModeFunction(
      "getattr", ImmutableList.of("object", "name", "default"), 2, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast, Environment env)
        throws EvalException {
      Object obj = args[0];
      String name = cast(args[1], String.class, "name", ast.getLocation());
      Object result = DotExpression.eval(obj, name, ast.getLocation());
      if (result == null) {
        if (args[2] != null) {
          return args[2];
        } else {
          throw new EvalException(ast.getLocation(), "Object of type '"
              + EvalUtils.getDataTypeName(obj) + "' has no field '" + name + "'");
        }
      }
      return result;
    }
  };

  @SkylarkBuiltin(name = "dir", returnType = SkylarkList.class,
      doc = "Returns a list strings: the names of the fields and "
          + "methods of the parameter object.",
      mandatoryParams = {@Param(name = "object", doc = "The object to check.")})
  private static final Function dir = new MixedModeFunction(
      "dir", ImmutableList.of("object"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast, Environment env)
        throws EvalException {
      Object obj = args[0];
      // Order the fields alphabetically.
      Set<String> fields = new TreeSet<>();
      if (obj instanceof ClassObject) {
        fields.addAll(((ClassObject) obj).getKeys());
      }
      fields.addAll(env.getFunctionNames(obj.getClass()));
      try {
        fields.addAll(FuncallExpression.getMethodNames(obj.getClass()));
      } catch (ExecutionException e) {
        // This shouldn't happen
        throw new EvalException(ast.getLocation(), e.getMessage());
      }
      return SkylarkList.list(fields, String.class);
    }
  };

  @SkylarkBuiltin(name = "type", returnType = String.class,
      doc = "Returns the type name of its argument.",
      mandatoryParams = {@Param(name = "object", doc = "The object to check type of.")})
  private static final Function type = new MixedModeFunction("type",
      ImmutableList.of("object"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast) throws EvalException {
      // There is no 'type' type in Skylark, so we return a string with the type name.
      return EvalUtils.getDataTypeName(args[0], false);
    }
  };

  @SkylarkBuiltin(name = "fail",
      doc = "Raises an error (the execution stops), except if the <code>when</code> condition "
      + "is False.",
      returnType = Environment.NoneType.class,
      mandatoryParams = {
        @Param(name = "msg", type = String.class, doc = "Error message to display for the user")},
      optionalParams = {
        @Param(name = "attr", type = String.class,
            doc = "The name of the attribute that caused the error"),
        @Param(name = "when", type = Boolean.class,
            doc = "When False, the function does nothing. Default is True.")})
  private static final Function fail = new MixedModeFunction(
      "fail", ImmutableList.of("msg", "attr", "when"), 1, false) {
    @Override
    public Object call(Object[] args, FuncallExpression ast, Environment env)
        throws EvalException {
      if (args[2] != null) {
        if (!EvalUtils.toBoolean(args[2])) {
          return Environment.NONE;
        }
      }
      String msg = cast(args[0], String.class, "msg", ast.getLocation());
      if (args[1] != null) {
        msg = "attribute " + cast(args[1], String.class, "attr", ast.getLocation())
            + ": " + msg;
      }
      throw new EvalException(ast.getLocation(), msg);
    }
  };

  @SkylarkBuiltin(name = "print", returnType = Environment.NoneType.class,
      doc = "Prints <code>msg</code> to the console.",
      mandatoryParams = {
      @Param(name = "*args", doc = "The objects to print.")},
      optionalParams = {
      @Param(name = "sep", type = String.class,
          doc = "The separator string between the objects, default is space (\" \").")})
  private static final Function print = new AbstractFunction("print") {
    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      String sep = " ";
      int count = 0;
      if (kwargs.containsKey("sep")) {
        sep = cast(kwargs.get("sep"), String.class, "sep", ast.getLocation());
        count = 1;
      }
      if (kwargs.size() > count) {
        kwargs = new HashMap<>(kwargs);
        kwargs.remove("sep");
        List<String> bad = Ordering.natural().sortedCopy(kwargs.keySet());
        throw new EvalException(ast.getLocation(), "unexpected keywords: '" + bad + "'");
      }
      String msg = Joiner.on(sep).join(Iterables.transform(args,
          new com.google.common.base.Function<Object, String>() {
        @Override
        public String apply(Object input) {
          return EvalUtils.printValue(input);
        }
      }));
      ((SkylarkEnvironment) env).handleEvent(Event.warn(ast.getLocation(), msg));
      return Environment.NONE;
    }
  };

  @SkylarkBuiltin(name = "zip",
      doc = "Returns a <code>list</code> of <code>tuple</code>s, where the i-th tuple contains "
          + "the i-th element from each of the argument sequences or iterables. The list has the "
          + "size of the shortest input. With a single iterable argument, it returns a list of "
          + "1-tuples. With no arguments, it returns an empty list. Examples:"
          + "<pre class=\"language-python\">"
          + "zip()  # == []\n"
          + "zip([1, 2])  # == [(1,), (2,)]\n"
          + "zip([1, 2], [3, 4])  # == [(1, 3), (2, 4)]\n"
          + "zip([1, 2], [3, 4, 5])  # == [(1, 3), (2, 4)]</pre>",
      returnType = SkylarkList.class)
  private static final Function zip = new AbstractFunction("zip") {
    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      Iterator<?>[] iterators = new Iterator<?>[args.size()];
      for (int i = 0; i < args.size(); i++) {
        iterators[i] = EvalUtils.toIterable(args.get(i), ast.getLocation()).iterator();
      }
      List<SkylarkList> result = new ArrayList<SkylarkList>();
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
          result.add(SkylarkList.tuple(elem));
        }
      } while (allHasNext);
      return SkylarkList.list(result, ast.getLocation());
    }
  };

  /**
   * Skylark String module.
   */
  @SkylarkModule(name = "string", doc =
      "A language built-in type to support strings. "
      + "Examples of string literals:<br>"
      + "<pre class=\"language-python\">a = 'abc\\ndef'\n"
      + "b = \"ab'cd\"\n"
      + "c = \"\"\"multiline string\"\"\"\n"
      + "\n"
      + "# Strings support slicing (negative index starts from the end):\n"
      + "x = \"hello\"[2:4]  # \"ll\"\n"
      + "y = \"hello\"[1:-1]  # \"ell\"\n"
      + "z = \"hello\"[:4]  # \"hell\"</pre>"
      + "Strings are iterable and support the <code>in</code> operator. Examples:<br>"
      + "<pre class=\"language-python\">\"a\" in \"abc\"   # evaluates as True\n"
      + "x = []\n"
     + "for s in \"abc\":\n"
      + "  x += [s]     # x == [\"a\", \"b\", \"c\"]</pre>")
  public static final class StringModule {}

  /**
   * Skylark Dict module.
   */
  @SkylarkModule(name = "dict", doc =
      "A language built-in type to support dicts. "
      + "Example of dict literal:<br>"
      + "<pre class=\"language-python\">d = {\"a\": 2, \"b\": 5}</pre>"
      + "Use brackets to access elements:<br>"
      + "<pre class=\"language-python\">e = d[\"a\"]   # e == 2</pre>"
      + "Dicts support the <code>+</code> operator to concatenate two dicts. In case of multiple "
      + "keys the second one overrides the first one. Examples:<br>"
      + "<pre class=\"language-python\">"
      + "d = {\"a\" : 1} + {\"b\" : 2}   # d == {\"a\" : 1, \"b\" : 2}\n"
      + "d += {\"c\" : 3}              # d == {\"a\" : 1, \"b\" : 2, \"c\" : 3}\n"
      + "d = d + {\"c\" : 5}           # d == {\"a\" : 1, \"b\" : 2, \"c\" : 5}</pre>"
      + "Since the language doesn't have mutable objects <code>d[\"a\"] = 5</code> automatically "
      + "translates to <code>d = d + {\"a\" : 5}</code>.<br>"
      + "Iterating on a dict is equivalent to iterating on its keys (in sorted order).<br>"
      + "Dicts support the <code>in</code> operator, testing membership in the keyset of the dict. "
      + "Example:<br>"
      + "<pre class=\"language-python\">\"a\" in {\"a\" : 2, \"b\" : 5}   # evaluates as True"
      + "</pre>")
  public static final class DictModule {}

  public static final List<Function> stringFunctions = ImmutableList.of(
      count, endswith, find, index, join, lower, replace, rfind,
      rindex, slice, split, startswith, strip, upper);

  public static final List<Function> listPureFunctions = ImmutableList.of(slice);

  public static final List<Function> listFunctions = ImmutableList.of(append, extend);

  public static final List<Function> dictFunctions = ImmutableList.of(items, get, keys, values);

  private static final List<Function> pureGlobalFunctions =
      ImmutableList.of(bool, int_, len, minus, select, sorted, str);

  private static final List<Function> skylarkGlobalFunctions = ImmutableList
      .<Function>builder()
      .addAll(pureGlobalFunctions)
      .add(list)
      .add(struct)
      .add(hasattr)
      .add(getattr)
      .add(set)
      .add(dir)
      .add(enumerate)
      .add(range)
      .add(type)
      .add(fail)
      .add(print)
      .add(zip)
      .build();

  /**
   * Set up a given environment for supported class methods.
   */
  public static void setupMethodEnvironment(Environment env) {
    env.registerFunction(Map.class, indexOperator.getName(), indexOperator);
    setupMethodEnvironment(env, Map.class, dictFunctions);
    env.registerFunction(String.class, indexOperator.getName(), indexOperator);
    setupMethodEnvironment(env, String.class, stringFunctions);
    setupMethodEnvironment(env, List.class, listPureFunctions);
    setupMethodEnvironment(env, SkylarkList.class, listPureFunctions);
    if (env.isSkylarkEnabled()) {
      env.registerFunction(SkylarkList.class, indexOperator.getName(), indexOperator);
      setupMethodEnvironment(env, skylarkGlobalFunctions);
    } else {
      env.registerFunction(List.class, indexOperator.getName(), indexOperator);
      env.registerFunction(ImmutableList.class, indexOperator.getName(), indexOperator);
      // TODO(bazel-team): listFunctions are not allowed in Skylark extensions (use += instead).
      // It is allowed in BUILD files only for backward-compatibility.
      setupMethodEnvironment(env, List.class, listFunctions);
      setupMethodEnvironment(env, stringFunctions);
      setupMethodEnvironment(env, pureGlobalFunctions);
    }
  }

  private static void setupMethodEnvironment(
      Environment env, Class<?> nameSpace, Iterable<Function> functions) {
    for (Function function : functions) {
      env.registerFunction(nameSpace, function.getName(), function);
    }
  }

  private static void setupMethodEnvironment(Environment env, Iterable<Function> functions) {
    for (Function function : functions) {
      env.update(function.getName(), function);
    }
  }

  /**
   * Collect global functions for the validation environment.
   */
  public static void setupValidationEnvironment(Set<String> builtIn) {
    for (Function function : skylarkGlobalFunctions) {
      builtIn.add(function.getName());
    }
  }
}
