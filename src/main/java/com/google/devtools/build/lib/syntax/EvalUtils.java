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
package com.google.devtools.build.lib.syntax;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Formattable;
import java.util.Formatter;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.Map;
import java.util.MissingFormatWidthException;
import java.util.Set;

/**
 * Utilities used by the evaluator.
 */
public abstract class EvalUtils {

  // TODO(bazel-team): Yet an other hack committed in the name of Skylark. One problem is that the
  // syntax package cannot depend on actions so we have to have this until Actions are immutable.
  // The other is that BuildConfigurations are technically not immutable but they cannot be modified
  // from Skylark.
  private static final ImmutableSet<Class<?>> quasiImmutableClasses;
  static {
    try {
      ImmutableSet.Builder<Class<?>> builder = ImmutableSet.builder();
      builder.add(Class.forName("com.google.devtools.build.lib.actions.Action"));
      builder.add(Class.forName("com.google.devtools.build.lib.analysis.config.BuildConfiguration"));
      builder.add(Class.forName("com.google.devtools.build.lib.actions.Root"));
      quasiImmutableClasses = builder.build();
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }

  private EvalUtils() {
  }

  /**
   * @return true if the specified sequence is a tuple; false if it's a modifiable list.
   */
  public static boolean isTuple(List<?> l) {
    return isTuple(l.getClass());
  }

  public static boolean isTuple(Class<?> c) {
    Preconditions.checkState(List.class.isAssignableFrom(c));
    return ImmutableList.class.isAssignableFrom(c);
  }

  /**
   * @return true if the specified value is immutable (suitable for use as a
   *         dictionary key) according to the rules of the Build language.
   */
  public static boolean isImmutable(Object o) {
    if (o instanceof Map<?, ?> || o instanceof Function
        || o instanceof FilesetEntry || o instanceof GlobList<?>) {
      return false;
    } else if (o instanceof List<?>) {
      return isTuple((List<?>) o); // tuples are immutable, lists are not.
    } else {
      return true; // string/int
    }
  }

  /**
   * Returns true if the type is immutable in the skylark language.
   */
  public static boolean isSkylarkImmutable(Class<?> c) {
    if (c.isAnnotationPresent(Immutable.class)) {
      return true;
    } else if (c.equals(String.class) || c.equals(Integer.class) || c.equals(Boolean.class)
        || SkylarkList.class.isAssignableFrom(c) || ImmutableMap.class.isAssignableFrom(c)
        || NestedSet.class.isAssignableFrom(c)) {
      return true;
    } else {
      for (Class<?> classObject : quasiImmutableClasses) {
        if (classObject.isAssignableFrom(c)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Returns a transitive superclass or interface implemented by c which is annotated
   * with SkylarkModule. Returns null if no such class or interface exists.
   */
  @VisibleForTesting
  static Class<?> getParentWithSkylarkModule(Class<?> c) {
    if (c == null) {
      return null;
    }
    if (c.isAnnotationPresent(SkylarkModule.class)) {
      return c;
    }
    Class<?> parent = getParentWithSkylarkModule(c.getSuperclass());
    if (parent != null) {
      return parent;
    }
    for (Class<?> ifparent : c.getInterfaces()) {
      ifparent = getParentWithSkylarkModule(ifparent);
      if (ifparent != null) {
        return ifparent;
      }
    }
    return null;
  }

  // TODO(bazel-team): move the following few type-related functions to SkylarkType
  /**
   * Compute the super-class of a class that Skylark considers as the Skylark type of its instances
   *
   * <p>Skylark type validation isn't otherwise equipped to deal with inheritance, so we must tell
   * it which is the super-class or interface that matters for Skylark type compatibility.
   * e.g. instances of all subclasses of SkylarkList are considered as being of type SkylarkList.
   *
   * @param c a class
   * @return a super-class of c to be used in validation-time type inference.
   */
  public static Class<?> getSkylarkType(Class<?> c) {
    if (ImmutableList.class.isAssignableFrom(c)) {
      return ImmutableList.class;
    } else if (List.class.isAssignableFrom(c)) {
      return List.class;
    } else if (SkylarkList.class.isAssignableFrom(c)) {
      return SkylarkList.class;
    } else if (Map.class.isAssignableFrom(c)) {
      return Map.class;
    } else if (NestedSet.class.isAssignableFrom(c)) {
      // This could be removed probably
      return NestedSet.class;
    } else if (Set.class.isAssignableFrom(c)) {
      return Set.class;
    } else {
      // TODO(bazel-team): also unify all ClassObject, that we print the same?
      //
      // Check if one of the superclasses or implemented interfaces has the SkylarkModule
      // annotation. If yes return that class.
      Class<?> parent = getParentWithSkylarkModule(c);
      if (parent != null) {
        return parent;
      }
    }
    return c;
  }

  /**
   * Returns a pretty name for the datatype of object 'o' in the Build language.
   */
  public static String getDataTypeName(Object o) {
    return getDataTypeName(o, false);
  }

  /**
   * Returns a pretty name for the datatype of object {@code object} in Skylark
   * or the BUILD language, with full details if the {@code full} boolean is true.
   */
  public static String getDataTypeName(Object object, boolean full) {
    Preconditions.checkNotNull(object);
    if (object instanceof SkylarkList) {
      SkylarkList list = (SkylarkList) object;
      if (list.isTuple()) {
        return "tuple";
      } else {
        return "list" + (full ? " of " + list.getContentType() + "s" : "");
      }
    } else if (object instanceof SkylarkNestedSet) {
      SkylarkNestedSet set = (SkylarkNestedSet) object;
      return "set" + (full ? " of " + set.getContentType() + "s" : "");
    } else {
      return getDataTypeNameFromClass(object.getClass());
    }
  }

  /**
   * Returns a pretty name for the datatype equivalent of class 'c' in the Build language.
   */
  public static String getDataTypeNameFromClass(Class<?> c) {
    if (c.equals(Object.class)) {
      return "unknown";
    } else if (c.equals(String.class)) {
      return "string";
    } else if (c.equals(Integer.class)) {
      return "int";
    } else if (c.equals(Boolean.class)) {
      return "bool";
    } else if (c.equals(Void.TYPE) || c.equals(Environment.NoneType.class)) {
      return "None";
    } else if (List.class.isAssignableFrom(c)) {
      // NB: the capital here is a subtle way to distinguish java Tuple and java List
      // from native SkylarkList tuple and list.
      // TODO(bazel-team): refactor SkylarkList and use it everywhere.
      return isTuple(c) ? "Tuple" : "List";
    } else if (GlobList.class.isAssignableFrom(c)) {
      return "glob list";
    } else if (Map.class.isAssignableFrom(c)) {
      return "dict";
    } else if (Function.class.isAssignableFrom(c)) {
      return "function";
    } else if (c.equals(FilesetEntry.class)) {
      return "FilesetEntry";
    } else if (NestedSet.class.isAssignableFrom(c) || SkylarkNestedSet.class.isAssignableFrom(c)) {
      return "set";
    } else if (ClassObject.SkylarkClassObject.class.isAssignableFrom(c)) {
      return "struct";
    } else if (SkylarkList.class.isAssignableFrom(c)) {
      // TODO(bazel-team): Refactor the class hierarchy so we can distinguish list and tuple types.
      return "list";
    } else if (c.isAnnotationPresent(SkylarkModule.class)) {
      SkylarkModule module = c.getAnnotation(SkylarkModule.class);
      return c.getAnnotation(SkylarkModule.class).name()
          + (module.namespace() ? " (a language module)" : "");
    } else {
      if (c.getSimpleName().isEmpty()) {
        return c.getName();
      } else {
        return c.getSimpleName();
      }
    }
  }

  /**
   * Returns a sequence of the appropriate list/tuple datatype for 'seq', based on 'isTuple'.
   */
  public static List<?> makeSequence(List<?> seq, boolean isTuple) {
    return isTuple ? ImmutableList.copyOf(seq) : seq;
  }

  /**
   * Print build-language value 'o' in display format into the specified buffer.
   */
  public static void printValue(Object o, Appendable buffer) {
    // Exception-swallowing wrapper due to annoying Appendable interface.
    try {
      printValueX(o, buffer);
    } catch (IOException e) {
      throw new AssertionError(e); // can't happen
    }
  }

  private static void printValueX(Object o, Appendable buffer)
      throws IOException {
    if (o == null) {
      throw new NullPointerException(); // None is not a build language value.
    } else if (o instanceof String || o instanceof Integer || o instanceof Double) {
      buffer.append(o.toString());

    } else if (o == Environment.NONE) {
      buffer.append("None");

    } else if (o == Boolean.TRUE) {
      buffer.append("True");

    } else if (o == Boolean.FALSE) {
      buffer.append("False");

    } else if (o instanceof List<?>) {
      List<?> seq = (List<?>) o;
      printList(seq, isImmutable(seq), buffer);

    } else if (o instanceof SkylarkList) {
      SkylarkList list = (SkylarkList) o;
      printList(list.toList(), list.isTuple(), buffer);

    } else if (o instanceof Map<?, ?>) {
      Map<?, ?> dict = (Map<?, ?>) o;
      printList(dict.entrySet(), "{", ", ", "}", buffer);

    } else if (o instanceof Map.Entry<?, ?>) {
      Map.Entry<?, ?> entry = (Map.Entry<?, ?>) o;
      prettyPrintValue(entry.getKey(), buffer);
      buffer.append(": ");
      prettyPrintValue(entry.getValue(), buffer);

    } else if (o instanceof SkylarkNestedSet) {
      Iterable<?> seq = (Iterable<?>) o;
      // TODO(bazel-team): preserve the order= flag when not the default "stable".
      printList(seq, "set([", ", ", "])", buffer);

    } else if (o instanceof Function) {
      Function func = (Function) o;
      buffer.append("<function " + func.getName() + ">");

    } else if (o instanceof FilesetEntry) {
      FilesetEntry entry = (FilesetEntry) o;
      buffer.append("FilesetEntry(srcdir = ");
      prettyPrintValue(entry.getSrcLabel().toString(), buffer);
      buffer.append(", files = ");
      prettyPrintValue(makeStringList(entry.getFiles()), buffer);
      buffer.append(", excludes = ");
      prettyPrintValue(makeList(entry.getExcludes()), buffer);
      buffer.append(", destdir = ");
      prettyPrintValue(entry.getDestDir().getPathString(), buffer);
      buffer.append(", strip_prefix = ");
      prettyPrintValue(entry.getStripPrefix(), buffer);
      buffer.append(", symlinks = \"");
      buffer.append(entry.getSymlinkBehavior().toString());
      buffer.append("\")");
    } else if (o instanceof PathFragment) {
      buffer.append(((PathFragment) o).getPathString());
    } else {
      buffer.append(o.toString());
    }
  }

  private static void printList(Iterable<?> list,
      String before, String separator, String after, Appendable buffer) throws IOException {
    String sep = "";
    buffer.append(before);
    for (Object o : list) {
      buffer.append(sep);
      prettyPrintValue(o, buffer);
      sep = separator;
    }
    buffer.append(after);
  }

  private static void printList(Iterable<?> list, boolean isTuple, Appendable buffer)
      throws IOException {
    if (isTuple) {
      printList(list, "(", ", ", ")", buffer);
    } else {
      printList(list, "[", ", ", "]", buffer);
    }
  }

  private static List<?> makeList(Collection<?> list) {
    return list == null ? Lists.newArrayList() : Lists.newArrayList(list);
  }

  private static List<String> makeStringList(List<Label> labels) {
    if (labels == null) { return Collections.emptyList(); }
    List<String> strings = Lists.newArrayListWithCapacity(labels.size());
    for (Label label : labels) {
      strings.add(label.toString());
    }
    return strings;
  }

  /**
   * Print build-language value 'o' in parseable format into the specified
   * buffer. (Only differs from printValueX in treatment of strings at toplevel,
   * i.e. not within a sequence or dict)
   */
  public static void prettyPrintValue(Object o, Appendable buffer) {
    // Exception-swallowing wrapper due to annoying Appendable interface.
    try {
      prettyPrintValueX(o, buffer);
    } catch (IOException e) {
      throw new AssertionError(e); // can't happen
    }
  }

  private static void prettyPrintValueX(Object o, Appendable buffer)
      throws IOException {
    if (o instanceof Label) {
      o = o.toString();  // Pretty-print a label like a string
    }
    if (o instanceof String) {
      String s = (String) o;
      buffer.append('"');
      for (int ii = 0, len = s.length(); ii < len; ++ii) {
        char c = s.charAt(ii);
        switch (c) {
        case '\r':
          buffer.append('\\').append('r');
          break;
        case '\n':
          buffer.append('\\').append('n');
          break;
        case '\t':
          buffer.append('\\').append('t');
          break;
        case '\"':
          buffer.append('\\').append('"');
          break;
        default:
          if (c < 32) {
            buffer.append(String.format("\\x%02x", (int) c));
          } else {
            buffer.append(c); // no need to support UTF-8
          }
        } // endswitch
      }
      buffer.append('\"');
    } else {
      printValueX(o, buffer);
    }
  }

  /**
   * Pretty-print value 'o' to a string. Convenience overloading of
   * prettyPrintValue(Object, Appendable).
   */
  public static String prettyPrintValue(Object o) {
    StringBuilder buffer = new StringBuilder();
    prettyPrintValue(o, buffer);
    return buffer.toString();
  }

  /**
   * Pretty-print values of 'o' separated by the separator.
   */
  public static String prettyPrintValues(String separator, Iterable<Object> o) {
    return Joiner.on(separator).join(Iterables.transform(o,
        new com.google.common.base.Function<Object, String>() {
      @Override
      public String apply(Object input) {
        return prettyPrintValue(input);
      }
    }));
  }

  /**
   * Print value 'o' to a string. Convenience overloading of printValue(Object, Appendable).
   */
  public static String printValue(Object o) {
    StringBuilder buffer = new StringBuilder();
    printValue(o, buffer);
    return buffer.toString();
  }

  public static Object checkNotNull(Expression expr, Object obj) throws EvalException {
    if (obj == null) {
      throw new EvalException(expr.getLocation(),
          "Unexpected null value, please send a bug report. "
          + "This was generated by '" + expr + "'");
    }
    return obj;
  }

  /**
   * Convert BUILD language objects to Formattable so JDK can render them correctly.
   * Don't do this for numeric or string types because we want %d, %x, %s to work.
   */
  private static Object makeFormattable(final Object o) {
    if (o instanceof Integer || o instanceof Double || o instanceof String) {
      return o;
    } else {
      return new Formattable() {
        @Override
        public String toString() {
          return "Formattable[" + o + "]";
        }

        @Override
        public void formatTo(Formatter formatter, int flags, int width,
            int precision) {
          printValue(o, formatter.out());
        }
      };
    }
  }

  private static final Object[] EMPTY = new Object[0];

  /*
   * N.B. MissingFormatWidthException is the only kind of IllegalFormatException
   * whose constructor can take and display arbitrary error message, hence its use below.
   */

  /**
   * Perform Python-style string formatting. Implemented by delegation to Java's
   * own string formatting routine to avoid reinventing the wheel. In more
   * obscure cases, semantics follow JDK (not Python) rules.
   *
   * @param pattern a format string.
   * @param tuple a tuple containing positional arguments
   */
  public static String formatString(String pattern, List<?> tuple)
      throws IllegalFormatException {
    int count = countPlaceholders(pattern);
    if (count < tuple.size()) {
      throw new MissingFormatWidthException(
          "not all arguments converted during string formatting");
    }

    List<Object> args = new ArrayList<>();

    for (Object o : tuple) {
      args.add(makeFormattable(o));
    }

    try {
      return String.format(pattern, args.toArray(EMPTY));
    } catch (IllegalFormatException e) {
      throw new MissingFormatWidthException(
          "invalid arguments for format string");
    }
  }

  private static int countPlaceholders(String pattern) {
    int length = pattern.length();
    boolean afterPercent = false;
    int i = 0;
    int count = 0;
    while (i < length) {
      switch (pattern.charAt(i)) {
        case 's':
        case 'd':
          if (afterPercent) {
            count++;
            afterPercent = false;
          }
          break;

        case '%':
          afterPercent = !afterPercent;
          break;

        default:
          if (afterPercent) {
            throw new MissingFormatWidthException("invalid arguments for format string");
          }
          afterPercent = false;
          break;
      }
      i++;
    }

    return count;
  }

  /**
   * @return the truth value of an object, according to Python rules.
   * http://docs.python.org/2/library/stdtypes.html#truth-value-testing
   */
  public static boolean toBoolean(Object o) {
    if (o == null || o == Environment.NONE) {
      return false;
    } else if (o instanceof Boolean) {
      return (Boolean) o;
    } else if (o instanceof String) {
      return !((String) o).isEmpty();
    } else if (o instanceof Integer) {
      return (Integer) o != 0;
    } else if (o instanceof Collection<?>) {
      return !((Collection<?>) o).isEmpty();
    } else if (o instanceof Map<?, ?>) {
      return !((Map<?, ?>) o).isEmpty();
    } else if (o instanceof NestedSet<?>) {
      return !((NestedSet<?>) o).isEmpty();
    } else if (o instanceof SkylarkNestedSet) {
      return !((SkylarkNestedSet) o).isEmpty();
    } else if (o instanceof Iterable<?>) {
      return !(Iterables.isEmpty((Iterable<?>) o));
    } else {
      return true;
    }
  }

  @SuppressWarnings("unchecked")
  public static Collection<Object> toCollection(Object o, Location loc) throws EvalException {
    if (o instanceof Collection) {
      return (Collection<Object>) o;
    } else if (o instanceof Map<?, ?>) {
      // For dictionaries we iterate through the keys only
      return ((Map<Object, Object>) o).keySet();
    } else if (o instanceof SkylarkNestedSet) {
      return ((SkylarkNestedSet) o).toCollection();
    } else {
      throw new EvalException(loc,
          "type '" + getDataTypeName(o) + "' is not a collection");
    }
  }

  @SuppressWarnings("unchecked")
  public static Iterable<Object> toIterable(Object o, Location loc) throws EvalException {
    if (o instanceof String) {
      // This is not as efficient as special casing String in for and dict and list comprehension
      // statements. However this is a more unified way.
      // The regex matches every character in the string until the end of the string,
      // so "abc" will be split into ["a", "b", "c"].
      return ImmutableList.<Object>copyOf(((String) o).split("(?!^)"));
    } else if (o instanceof Iterable) {
      return (Iterable<Object>) o;
    } else if (o instanceof Map<?, ?>) {
      // For dictionaries we iterate through the keys only
      return ((Map<Object, Object>) o).keySet();
    } else {
      throw new EvalException(loc,
          "type '" + getDataTypeName(o) + "' is not an iterable");
    }
  }

  /**
   * Returns the size of the Skylark object or -1 in case the object doesn't have a size.
   */
  public static int size(Object arg) {
    if (arg instanceof String) {
      return ((String) arg).length();
    } else if (arg instanceof Map) {
      return ((Map<?, ?>) arg).size();
    } else if (arg instanceof SkylarkList) {
      return ((SkylarkList) arg).size();
    } else if (arg instanceof Iterable) {
      // Iterables.size() checks if arg is a Collection so it's efficient in that sense.
      return Iterables.size((Iterable<?>) arg);
    }
    return -1;
  }
}
