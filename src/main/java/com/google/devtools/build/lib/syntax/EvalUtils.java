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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import com.google.devtools.build.lib.syntax.compiler.ByteCodeUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

import net.bytebuddy.implementation.bytecode.StackManipulation;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Utilities used by the evaluator.
 */
public final class EvalUtils {

  private EvalUtils() {}

  /**
   * The exception that SKYLARK_COMPARATOR might throw. This is an unchecked exception
   * because Comparator doesn't let us declare exceptions. It should normally be caught
   * and wrapped in an EvalException.
   */
  public static class ComparisonException extends RuntimeException {
    public ComparisonException(String msg) {
      super(msg);
    }
  }

  /**
   * Compare two Skylark objects.
   *
   * <p> It may throw an unchecked exception ComparisonException that should be wrapped in
   * an EvalException.
   */
  public static final Ordering<Object> SKYLARK_COMPARATOR = new Ordering<Object>() {
    private int compareLists(SkylarkList o1, SkylarkList o2) {
      for (int i = 0; i < Math.min(o1.size(), o2.size()); i++) {
        int cmp = compare(o1.get(i), o2.get(i));
        if (cmp != 0) {
          return cmp;
        }
      }
      return Integer.compare(o1.size(), o2.size());
    }

    @Override
    @SuppressWarnings("unchecked")
    public int compare(Object o1, Object o2) {
      o1 = SkylarkType.convertToSkylark(o1, /*env=*/ null);
      o2 = SkylarkType.convertToSkylark(o2, /*env=*/ null);

      if (o1 instanceof SkylarkList && o2 instanceof SkylarkList
          && ((SkylarkList) o1).isTuple() == ((SkylarkList) o2).isTuple()) {
        return compareLists((SkylarkList) o1, (SkylarkList) o2);
      }
      try {
        return ((Comparable<Object>) o1).compareTo(o2);
      } catch (ClassCastException e) {
        try {
          // Different types -> let the class names decide
          return o1.getClass().getName().compareTo(o2.getClass().getName());
        } catch (NullPointerException ex) {
          throw new ComparisonException(
              "Cannot compare " + getDataTypeName(o1) + " with " + EvalUtils.getDataTypeName(o2));
        }
      }
    }
  };

  public static final StackManipulation checkValidDictKey =
      ByteCodeUtils.invoke(EvalUtils.class, "checkValidDictKey", Object.class);

  /**
   * Checks that an Object is a valid key for a Skylark dict.
   * @param o an Object to validate
   * @throws EvalException if o is not a valid key
   */
  public static void checkValidDictKey(Object o) throws EvalException {
    // TODO(bazel-team): check that all recursive elements are both Immutable AND Comparable.
    if (isImmutable(o)) {
      return;
    }
    // Same error message as Python (that makes it a TypeError).
    throw new EvalException(null, Printer.format("unhashable type: '%r'", o.getClass()));
  }

  /**
   * Is this object known or assumed to be recursively immutable by Skylark?
   * @param o an Object
   * @return true if the object is known to be an immutable value.
   */
  // NB: This is used as the basis for accepting objects in SkylarkNestedSet-s,
  // as well as for accepting objects as keys for Skylark dict-s.
  public static boolean isImmutable(Object o) {
    if (o instanceof Tuple) {
      for (Object item : (Tuple) o) {
        if (!isImmutable(item)) {
          return false;
        }
      }
      return true;
    }
    if (o instanceof SkylarkMutable) {
      return false;
    }
    if (o instanceof SkylarkValue) {
      return ((SkylarkValue) o).isImmutable();
    }
    return isImmutable(o.getClass());
  }

  /**
   * Is this class known to be *recursively* immutable by Skylark?
   * For instance, class Tuple is not it, because it can contain mutable values.
   * @param c a Class
   * @return true if the class is known to represent only recursively immutable values.
   */
  // NB: This is used as the basis for accepting objects in SkylarkNestedSet-s,
  // as well as for accepting objects as keys for Skylark dict-s.
  static boolean isImmutable(Class<?> c) {
    return c.isAnnotationPresent(Immutable.class) // TODO(bazel-team): beware of containers!
        || c.equals(String.class)
        || c.equals(Integer.class)
        || c.equals(Boolean.class);
  }

  /**
   * Returns true if the type is acceptable to be returned to the Skylark language.
   */
  public static boolean isSkylarkAcceptable(Class<?> c) {
    return SkylarkValue.class.isAssignableFrom(c) // implements SkylarkValue
        || c.equals(String.class) // basic values
        || c.equals(Integer.class)
        || c.equals(Boolean.class)
        || c.isAnnotationPresent(SkylarkModule.class) // registered Skylark class
        || ImmutableMap.class.isAssignableFrom(c) // will be converted to SkylarkDict
        || NestedSet.class.isAssignableFrom(c) // will be converted to SkylarkNestedSet
        || c.equals(PathFragment.class); // other known class
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
   * Return the Skylark-type of {@code c}
   *
   * <p>The result will be a type that Skylark understands and is either equal to {@code c}
   * or is a supertype of it. For example, all instances of (all subclasses of) SkylarkList
   * are considered to be SkylarkLists.
   *
   * <p>Skylark's type validation isn't equipped to deal with inheritance so we must tell it which
   * of the superclasses or interfaces of {@code c} is the one that matters for type compatibility.
   *
   * @param c a class
   * @return a super-class of c to be used in validation-time type inference.
   */
  public static Class<?> getSkylarkType(Class<?> c) {
    if (SkylarkList.class.isAssignableFrom(c)) {
      return c;
    } else if (ImmutableList.class.isAssignableFrom(c)) {
      return ImmutableList.class;
    } else if (List.class.isAssignableFrom(c)) {
      return List.class;
    } else if (Map.class.isAssignableFrom(c)) {
      return Map.class;
    } else if (NestedSet.class.isAssignableFrom(c)) {
      // This could be removed probably
      return NestedSet.class;
    } else if (Set.class.isAssignableFrom(c)) {
      return Set.class;
    } else {
      // TODO(bazel-team): also unify all implementations of ClassObject,
      // that we used to all print the same as "struct"?
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
  public static String getDataTypeName(Object object, boolean fullDetails) {
    Preconditions.checkNotNull(object);
    if (fullDetails) {
      if (object instanceof SkylarkNestedSet) {
        SkylarkNestedSet set = (SkylarkNestedSet) object;
        return "set of " + set.getContentType() + "s";
      }
      if (object instanceof SelectorList) {
        SelectorList list = (SelectorList) object;
        return "select of " + getDataTypeNameFromClass(list.getType());
      }
    }
    return getDataTypeNameFromClass(object.getClass());
  }

  /**
   * Returns a pretty name for the datatype equivalent of class 'c' in the Build language.
   */
  public static String getDataTypeNameFromClass(Class<?> c) {
    return getDataTypeNameFromClass(c, true);
  }

  /**
   * Returns a pretty name for the datatype equivalent of class 'c' in the Build language.
   * @param highlightNameSpaces Determines whether the result should also contain a special comment
   * when the given class identifies a Skylark name space.
   */
  public static String getDataTypeNameFromClass(Class<?> c, boolean highlightNameSpaces) {
    if (c.isAnnotationPresent(SkylarkModule.class)) {
      SkylarkModule module = c.getAnnotation(SkylarkModule.class);
      return c.getAnnotation(SkylarkModule.class).name()
          + ((module.namespace() && highlightNameSpaces) ? " (a language module)" : "");
    } else if (c.equals(Object.class)) {
      return "unknown";
    } else if (c.equals(String.class)) {
      return "string";
    } else if (c.equals(Integer.class)) {
      return "int";
    } else if (c.equals(Boolean.class)) {
      return "bool";
    } else if (Map.class.isAssignableFrom(c)) {
      return "dict";
    } else if (BaseFunction.class.isAssignableFrom(c)) {
      return "function";
    } else if (c.equals(SelectorValue.class)) {
      return "select";
    } else if (NestedSet.class.isAssignableFrom(c) || SkylarkNestedSet.class.isAssignableFrom(c)) {
      // TODO(bazel-team): no one should be seeing naked NestedSet at all.
      return "set";
    } else if (ClassObject.SkylarkClassObject.class.isAssignableFrom(c)) {
      return "struct";
    } else {
      if (c.getSimpleName().isEmpty()) {
        return c.getName();
      } else {
        return c.getSimpleName();
      }
    }
  }

  public static Object checkNotNull(Expression expr, Object obj) throws EvalException {
    if (obj == null) {
      throw new EvalException(expr.getLocation(),
          "Unexpected null value, please send a bug report. "
          + "This was generated by '" + expr + "'");
    }
    return obj;
  }

  public static final StackManipulation toBoolean =
      ByteCodeUtils.invoke(EvalUtils.class, "toBoolean", Object.class);

  /**
   * @return the truth value of an object, according to Python rules.
   * http://docs.python.org/2/library/stdtypes.html#truth-value-testing
   */
  public static boolean toBoolean(Object o) {
    if (o == null || o == Runtime.NONE) {
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

  public static final StackManipulation toCollection =
      ByteCodeUtils.invoke(EvalUtils.class, "toCollection", Object.class, Location.class);

  public static Collection<?> toCollection(Object o, Location loc) throws EvalException {
    if (o instanceof Collection) {
      return (Collection<?>) o;
    } else if (o instanceof SkylarkList) {
      return ((SkylarkList) o).getImmutableList();
    } else if (o instanceof Map) {
      // For dictionaries we iterate through the keys only
      // For determinism, we sort the keys.
      try {
        return SKYLARK_COMPARATOR.sortedCopy(((Map<?, ?>) o).keySet());
      } catch (ComparisonException e) {
        throw new EvalException(loc, e);
      }
    } else if (o instanceof SkylarkNestedSet) {
      return ((SkylarkNestedSet) o).toCollection();
    } else {
      throw new EvalException(loc,
          "type '" + getDataTypeName(o) + "' is not a collection");
    }
  }

  public static final StackManipulation toIterable =
      ByteCodeUtils.invoke(EvalUtils.class, "toIterable", Object.class, Location.class);

  public static Iterable<?> toIterable(Object o, Location loc) throws EvalException {
    if (o instanceof String) {
      // This is not as efficient as special casing String in for and dict and list comprehension
      // statements. However this is a more unified way.
      return split((String) o);
    } else if (o instanceof Iterable) {
      return (Iterable<?>) o;
    } else if (o instanceof Map) {
      return toCollection(o, loc);
    } else {
      throw new EvalException(loc,
          "type '" + getDataTypeName(o) + "' is not iterable");
    }
  }

  private static ImmutableList<String> split(String value) {
    ImmutableList.Builder<String> builder = new ImmutableList.Builder<>();
    for (char c : value.toCharArray()) {
      builder.add(String.valueOf(c));
    }
    return builder.build();
  }

  /**
   * @return the size of the Skylark object or -1 in case the object doesn't have a size.
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

  /** @return true if x is Java null or Skylark None */
  public static boolean isNullOrNone(Object x) {
    return x == null || x == Runtime.NONE;
  }

  /**
   * Build a map of kwarg arguments from a list, removing null-s or None-s.
   *
   * @param init a series of key, value pairs (as consecutive arguments)
   *   as in {@code optionMap(k1, v1, k2, v2, k3, v3)}
   *   where each key is a String, each value is an arbitrary Objet.
   * @return a {@code Map<String, Object>} that has all the specified entries,
   *   where key, value pairs appearing earlier have precedence,
   *   i.e. {@code k1, v1} may override {@code k3, v3}.
   *
   * Ignore any entry where the value is null or None.
   * Keys cannot be null.
   */
  @SuppressWarnings("unchecked")
  public static ImmutableMap<String, Object> optionMap(Object... init) {
    ImmutableMap.Builder<String, Object> b = new ImmutableMap.Builder<>();
    Preconditions.checkState(init.length % 2 == 0);
    for (int i = init.length - 2; i >= 0; i -= 2) {
      String key = (String) Preconditions.checkNotNull(init[i]);
      Object value = init[i + 1];
      if (!isNullOrNone(value)) {
        b.put(key, value);
      }
    }
    return b.build();
  }
}
