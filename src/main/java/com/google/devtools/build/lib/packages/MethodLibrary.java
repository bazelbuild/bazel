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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.syntax.AbstractFunction;
import com.google.devtools.build.lib.syntax.AbstractFunction.NoArgFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.FuncallExpression.MethodDescriptor;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.MixedModeFunction;
import com.google.devtools.build.lib.syntax.PositionalFunction;
import com.google.devtools.build.lib.syntax.PositionalFunction.SimplePositionalFunction;
import com.google.devtools.build.lib.syntax.SelectorValue;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.SkylarkType.SkylarkFunctionType;

import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
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

  @SkylarkBuiltin(name = "join", objectType = StringModule.class,
      doc = "Returns a string in which the string elements of the argument have been "
          + "joined by this string as a separator. Example:<br>"
          + "<pre class=code>\"|\".join([\"a\", \"b\", \"c\"]) == \"a|b|c\"</pre>")
  private static Function join = new SimplePositionalFunction("join", 2, 2) {
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

  @SkylarkBuiltin(name = "lower", objectType = StringModule.class,
      doc = "Returns the lower case version of this string.")
  private static Function lower = new SimplePositionalFunction("lower", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(args.get(0), "'lower' operand");
      return thiz.toLowerCase();
    }
  };

  @SkylarkBuiltin(name = "replace", objectType = StringModule.class,
      doc = "Returns a copy of the string in which the occurrences "
          + "of <code>old</code> have been replaced with <code>new</code>, optionally restricting "
          + "the number of replacements to <code>maxsplit</code>.")
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

  @SkylarkBuiltin(name = "split", objectType = StringModule.class,
      doc = "Returns a list of all the words in the string, using <code>sep</code>  "
          + "as the separator, optionally limiting the number of splits to <code>maxsplit</code>.")
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

  @SkylarkBuiltin(name = "rfind", objectType = StringModule.class,
      doc = "Returns the last index where <code>sub</code> is found, "
          + "or -1 if no such index exists, optionally restricting to "
          + "[<code>start</code>:<code>end</code>], "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.")
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

  @SkylarkBuiltin(name = "find", objectType = StringModule.class,
      doc = "Returns the first index where <code>sub</code> is found, "
          + "or -1 if no such index exists, optionally restricting to "
          + "[<code>start</code>:<code>end]</code>, "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.")
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

  @SkylarkBuiltin(name = "endswith", objectType = StringModule.class,
      doc = "Returns True if the string ends with <code>sub</code>, "
          + "otherwise False, optionally restricting to [<code>start</code>:<code>end</code>], "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.")
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

  @SkylarkBuiltin(name = "startswith", objectType = StringModule.class,
      doc = "Returns True if the string starts with <code>sub</code>, "
          + "otherwise False, optionally restricting to [<code>start</code>:<code>end</code>], "
          + "<code>start</code> being inclusive and <code>end</code> being exclusive.")
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
  @SkylarkBuiltin(name = "strip", objectType = StringModule.class,
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
  @SkylarkBuiltin(name = "substring", hidden = true,
      doc = "String[<code>start</code>:<code>end</code>] returns a substring.")
  private static Function substring = new SimplePositionalFunction("$substring", 3, 3) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
      String thiz = Type.STRING.convert(args.get(0), "substring operand");
      int left = Type.INTEGER.convert(args.get(1), "substring operand");
      int right = Type.INTEGER.convert(args.get(2), "substring operand");
      return getPythonSubstring(thiz, left, right);
    }
  };

  // supported list methods
  @SkylarkBuiltin(name = "append", hidden = true,
      doc = "Adds an item to the end of the list.")
  private static Function append = new SimplePositionalFunction("append", 2, 2) {
    // @SuppressWarnings("unchecked")
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException,
        ConversionException {
      List<Object> thiz = Type.OBJECT_LIST.convert(args.get(0), "'append' operand");
      thiz.add(args.get(1));
      return Environment.NONE;
    }
  };

  @SkylarkBuiltin(name = "extend", hidden = true,
      doc = "Adds all items to the end of the list.")
  private static Function extend = new SimplePositionalFunction("extend", 2, 2) {
    // @SuppressWarnings("unchecked")
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException,
        ConversionException {
      List<Object> thiz = Type.OBJECT_LIST.convert(args.get(0), "'extend' operand");
      List<Object> l = Type.OBJECT_LIST.convert(args.get(1), "'extend' argument");
      thiz.addAll(l);
      return Environment.NONE;
    }
  };

  // dictionary access operator
  @SkylarkBuiltin(name = "$index", hidden = true,
      doc = "Returns the nth element of a list or string, "
          + "or looks up a value in a dictionary.")
  private static Function index = new SimplePositionalFunction("$index", 2, 2) {
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
      } else {
        // TODO(bazel-team): This is dead code, get rid of it.
        throw new EvalException(ast.getLocation(), String.format(
            "Unsupported datatype (%s) for indexing, only works for dict and list",
            EvalUtils.getDatatypeName(collectionCandidate)));
      }
    }
  };

  @SkylarkBuiltin(name = "values", objectType = DictModule.class,
      doc = "Return the list of values.")
  private static Function values = new NoArgFunction("values") {
    @Override
    public Object call(Object self, FuncallExpression ast, Environment env)
        throws EvalException, InterruptedException {
      Map<?, ?> dict = (Map<?, ?>) self;
      return convert(dict.values(), env, ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "items", objectType = DictModule.class,
      doc = "Return the list of key-value tuples.")
  private static Function items = new NoArgFunction("items") {
    @Override
    public Object call(Object self, FuncallExpression ast, Environment env)
        throws EvalException, InterruptedException {
      Map<?, ?> dict = (Map<?, ?>) self;
      List<Object> list = Lists.newArrayListWithCapacity(dict.size());
      for (Map.Entry<?, ?> entries : dict.entrySet()) {
        List<?> item = ImmutableList.of(entries.getKey(), entries.getValue());
        list.add(env.isSkylarkEnabled() ? SkylarkList.tuple(item) : item);
      }
      return convert(list, env, ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "keys", objectType = DictModule.class,
      doc = "Return the list of keys.")
  private static Function keys = new NoArgFunction("keys") {
    @Override
    public Object call(Object self, FuncallExpression ast, Environment env)
        throws EvalException, InterruptedException {
      Map<?, ?> dict = (Map<?, ?>) self;
      return convert(dict.keySet(), env, ast.getLocation());
    }
  };

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
  @SkylarkBuiltin(name = "-", hidden = true, doc = "Unary minus operator.")
  private static Function minus = new SimplePositionalFunction("-", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws ConversionException {
      int num = Type.INTEGER.convert(args.get(0), "'unary minus' argument");
      return -num;
    }
  };

  @SkylarkBuiltin(name = "list", doc = "Converts a collection (e.g. set or dictionary) to a list.")
  private static Function list = new SimplePositionalFunction("list", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException {
      Location loc = ast.getLocation();
      return SkylarkList.list(EvalUtils.toCollection(args.get(0), loc), loc);
    }
  };

  @SkylarkBuiltin(name = "len", doc =
      "Returns the length of a string, list, tuple, set, or dictionary.")
  private static Function len = new SimplePositionalFunction("len", 1, 1) {

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

  @SkylarkBuiltin(name = "str", doc =
      "Converts any object to string. This is useful for debugging.")
  private static Function str = new SimplePositionalFunction("str", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException {
      return EvalUtils.printValue(args.get(0));
    }
  };

  @SkylarkBuiltin(name = "bool", doc = "Converts an object to boolean. "
      + "It returns False if the object is None, False, an empty string, the number 0, or an "
      + "empty collection. Otherwise, it returns True.")
  private static Function bool = new SimplePositionalFunction("bool", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException {
      return EvalUtils.toBoolean(args.get(0));
    }
  };

  @SkylarkBuiltin(name = "struct", doc =
      "Creates an immutable struct using the keyword arguments as fields. It is used to group "
      + "multiple values together.Example:<br>"
      + "<pre class=code>s = struct(x = 2, y = 3)\n"
      + "return s.x + s.y  # returns 5</pre>")
  private static Function struct = new AbstractFunction("struct") {

    @Override
    public Object call(List<Object> args, Map<String, Object> kwargs, FuncallExpression ast,
        Environment env) throws EvalException, InterruptedException {
      if (args.size() > 0) {
        throw new EvalException(ast.getLocation(), "struct only supports keyword arguments");
      }
      return new SkylarkClassObject(kwargs, ast.getLocation());
    }
  };

  @SkylarkBuiltin(name = "set",
      doc = "Creates a set from the <code>items</code>, that supports nesting. "
          + "The nesting is applied to other nested sets among <code>items</code>. "
          + "Ordering can be: <code>stable</code> (default), <code>compile</code>, "
          + "<code>link</code> or <code>naive_link</code>.<br>"
          + "Examples:<br>"
          + "<pre class=code>set([1, set([2, 3]), 2])\n"
          + "set([1, 2, 3], order=\"compile\")</pre>")
  private static final Function set =
    new MixedModeFunction("set", ImmutableList.of("items", "order"), 0, false) {
    @Override
    public Object call(Object[] namedArguments, List<Object> positionalArguments,
        Map<String, Object> keywordArguments, FuncallExpression ast) throws EvalException,
        ConversionException {
      Order order;
      if (namedArguments[1] == null || namedArguments[1].equals("stable")) {
        order = Order.STABLE_ORDER;
      } else if (namedArguments[1].equals("compile")) {
        order = Order.COMPILE_ORDER;
      } else if (namedArguments[1].equals("link")) {
        order = Order.LINK_ORDER;
      } else if (namedArguments[1].equals("naive_link")) {
        order = Order.NAIVE_LINK_ORDER;
      } else {
        throw new EvalException(ast.getLocation(), "Invalid order: " + namedArguments[1]);
      }

      if (namedArguments[0] == null) {
        return new SkylarkNestedSet(order, SkylarkList.EMPTY_LIST, ast.getLocation());
      }
      return new SkylarkNestedSet(order, namedArguments[0], ast.getLocation());
    }
  };

  /**
   * Returns a function-value implementing "select" (i.e. configurable attributes)
   * in the specified package context.
   */
  @SkylarkBuiltin(name = "select",
      doc = "Creates a SelectorValue from the dict parameter.")
  private static final Function select = new SimplePositionalFunction("select", 1, 1) {
      @Override
      public Object call(List<Object> args, FuncallExpression ast)
          throws EvalException, ConversionException {
        Object dict = Iterables.getOnlyElement(args);
        if (!(dict instanceof Map<?, ?>)) {
          throw new EvalException(ast.getLocation(),
              "select({...}) argument isn't a dictionary");
        }
        return new SelectorValue((Map<?, ?>) dict);
      }
    };

  /**
   * Returns true if the struct has a field of the given name, otherwise false.
   */
  @SkylarkBuiltin(name = "hasattr",
      doc = "Returns True if the struct has a field of the given name, otherwise False. "
      + "Example:<br>"
      + "<pre class=code>hasattr(ctx.attr, \"myattr\")</pre>")
  private static final Function hasattr = new SimplePositionalFunction("hasattr", 2, 2) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast) throws EvalException,
        ConversionException {
      Object obj = args.get(0);
      String name = cast(args.get(1), String.class, "second param", ast.getLocation());
      if (obj instanceof ClassObject) {
        return ((ClassObject) obj).getValue(name) != null;
      } else {
        try {
          List<MethodDescriptor> methods = FuncallExpression.getMethods(obj.getClass(), name, 0);
          return methods != null && methods.size() > 0 && Iterables.getOnlyElement(methods)
              .getAnnotation().structField();
        } catch (ExecutionException e) {
          // This shouldn't happen
          throw new EvalException(ast.getLocation(), e.getMessage());
        }
      }
    }
  };

  @SkylarkBuiltin(name = "dir",
      doc = "Returns the list of fields and methods of the parameter object.")
  private static final Function dir = new PositionalFunction("dir", 1, 1) {
    @Override
    public Object call(List<Object> args, FuncallExpression ast, Environment env)
        throws EvalException {
      Object obj = args.get(0);
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

  /**
   * Skylark String module.
   */
  @SkylarkModule(name = "string", doc =
      "A language built-in type to support strings. "
      + "Example of string literals:<br>"
      + "<pre class=code>a = 'abc\\ndef'\n"
      + "b = \"ab'cd\"\n"
      + "c = \"\"\"multiline string\"\"\"</pre>")
  public static final class StringModule {}

  /**
   * Skylark Dict module.
   */
  @SkylarkModule(name = "dict", doc =
      "A language built-in type to support dicts. "
      + "Example of dict literal:<br>"
      + "<pre class=code>d = {\"a\": 2, \"b\": 5}</pre>")
  public static final class DictModule {}

  public static final Map<Function, SkylarkType> stringFunctions = ImmutableMap
      .<Function, SkylarkType>builder()
      .put(join, SkylarkType.STRING)
      .put(lower, SkylarkType.STRING)
      .put(replace, SkylarkType.STRING)
      .put(split, SkylarkType.of(List.class, String.class))
      .put(rfind, SkylarkType.INT)
      .put(find, SkylarkType.INT)
      .put(endswith, SkylarkType.BOOL)
      .put(startswith, SkylarkType.BOOL)
      .put(strip, SkylarkType.STRING)
      .put(substring, SkylarkType.STRING)
      .build();

  public static final List<Function> listFunctions = ImmutableList
      .<Function>builder()
      .add(append)
      .add(extend)
      .build();

  public static final Map<Function, SkylarkType> dictFunctions = ImmutableMap
      .<Function, SkylarkType>builder()
      .put(items, SkylarkType.of(List.class))
      .put(keys, SkylarkType.of(Set.class))
      .put(values, SkylarkType.of(List.class))
      .build();

  private static final Map<Function, SkylarkType> pureGlobalFunctions = ImmutableMap
      .<Function, SkylarkType>builder()
      // TODO(bazel-team): String methods are added two times, because there are
      // a lot of cases when they are used as global functions in the depot. Those
      // should be cleaned up first.
      .put(minus, SkylarkType.INT)
      .put(select, SkylarkType.of(SelectorValue.class))
      .put(len, SkylarkType.INT)
      .put(str, SkylarkType.STRING)
      .put(bool, SkylarkType.BOOL)
      .build();

  private static final Map<Function, SkylarkType> skylarkGlobalFunctions = ImmutableMap
      .<Function, SkylarkType>builder()
      .putAll(pureGlobalFunctions)
      .put(list, SkylarkType.of(SkylarkList.class))
      .put(struct, SkylarkType.of(ClassObject.class))
      .put(hasattr, SkylarkType.BOOL)
      .put(set, SkylarkType.of(SkylarkNestedSet.class))
      .put(dir, SkylarkType.of(SkylarkList.class, String.class))
      .build();

  /**
   * Set up a given environment for supported class methods.
   */
  public static void setupMethodEnvironment(Environment env) {
    env.registerFunction(Map.class, index.getName(), index);
    setupMethodEnvironment(env, Map.class, dictFunctions.keySet());
    setupMethodEnvironment(env, String.class, stringFunctions.keySet());
    if (env.isSkylarkEnabled()) {
      env.registerFunction(SkylarkList.class, index.getName(), index);
      setupMethodEnvironment(env, skylarkGlobalFunctions.keySet());
    } else {
      env.registerFunction(List.class, index.getName(), index);
      env.registerFunction(ImmutableList.class, index.getName(), index);
      // TODO(bazel-team): listFunctions are not allowed in Skylark extensions (use += instead).
      // It is allowed in BUILD files only for backward-compatibility.
      setupMethodEnvironment(env, List.class, listFunctions);
      setupMethodEnvironment(env, stringFunctions.keySet());
      setupMethodEnvironment(env, pureGlobalFunctions.keySet());
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

  private static void setupValidationEnvironment(
      Map<Function, SkylarkType> functions, Map<String, SkylarkType> result) {
    for (Map.Entry<Function, SkylarkType> function : functions.entrySet()) {
      String name = function.getKey().getName();
      result.put(name, SkylarkFunctionType.of(name, function.getValue()));
    }
  }

  public static void setupValidationEnvironment(
      Map<SkylarkType, Map<String, SkylarkType>> builtIn) {
    Map<String, SkylarkType> global = builtIn.get(SkylarkType.GLOBAL);
    setupValidationEnvironment(skylarkGlobalFunctions, global);

    Map<String, SkylarkType> dict = new HashMap<>();
    setupValidationEnvironment(dictFunctions, dict);
    builtIn.put(SkylarkType.of(Map.class), dict);

    Map<String, SkylarkType> string = new HashMap<>();
    setupValidationEnvironment(stringFunctions, string);
    builtIn.put(SkylarkType.STRING, string);
  }
}
