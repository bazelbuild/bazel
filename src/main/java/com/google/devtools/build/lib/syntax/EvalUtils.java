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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

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
   * <p>It may throw an unchecked exception ComparisonException that should be wrapped in an
   * EvalException.
   */
  public static final Ordering<Object> SKYLARK_COMPARATOR =
      new Ordering<Object>() {
        private int compareLists(SkylarkList o1, SkylarkList o2) {
          if (o1 instanceof RangeList || o2 instanceof RangeList) {
            throw new ComparisonException("Cannot compare range objects");
          }

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
          o1 = SkylarkType.convertToSkylark(o1, (Environment) null);
          o2 = SkylarkType.convertToSkylark(o2, (Environment) null);

          if (o1 instanceof SkylarkList
              && o2 instanceof SkylarkList
              && ((SkylarkList) o1).isTuple() == ((SkylarkList) o2).isTuple()) {
            return compareLists((SkylarkList) o1, (SkylarkList) o2);
          }

          if (!(o1.getClass().isAssignableFrom(o2.getClass())
              || o2.getClass().isAssignableFrom(o1.getClass()))) {
            throw new ComparisonException(
                "Cannot compare " + getDataTypeName(o1) + " with " + getDataTypeName(o2));
          }

          if (o1 instanceof ClassObject) {
            throw new ComparisonException("Cannot compare structs");
          }
          if (o1 instanceof SkylarkNestedSet) {
            throw new ComparisonException("Cannot compare depsets");
          }
          try {
            return ((Comparable<Object>) o1).compareTo(o2);
          } catch (ClassCastException e) {
            throw new ComparisonException(
                "Cannot compare " + getDataTypeName(o1) + " with " + getDataTypeName(o2));
          }
        }
      };

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
        // there is a registered Skylark ancestor class (useful e.g. when using AutoValue)
        || SkylarkInterfaceUtils.getSkylarkModule(c) != null
        || ImmutableMap.class.isAssignableFrom(c) // will be converted to SkylarkDict
        || NestedSet.class.isAssignableFrom(c) // will be converted to SkylarkNestedSet
        || PathFragment.class.isAssignableFrom(c); // other known class
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
    // TODO(bazel-team): Iterable and Class likely do not belong here.
    if (String.class.equals(c)
        || Boolean.class.equals(c)
        || Integer.class.equals(c)
        || Iterable.class.equals(c)
        || Class.class.equals(c)) {
      return c;
    }
    // TODO(bazel-team): We should require all Skylark-addressable values that aren't builtin types
    // (String/Boolean/Integer) to implement SkylarkValue. We should also require them to have a
    // (possibly inherited) @SkylarkModule annotation.
    Class<?> parent = SkylarkInterfaceUtils.getParentWithSkylarkModule(c);
    if (parent != null) {
      return parent;
    }
    Preconditions.checkArgument(
        SkylarkValue.class.isAssignableFrom(c),
        "%s is not allowed as a Starlark value (getSkylarkType() failed)",
        c);
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
        return "depset of " + set.getContentType() + "s";
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
    SkylarkModule module = SkylarkInterfaceUtils.getSkylarkModule(c);
    if (module != null) {
      return module.name()
          + ((module.namespace() && highlightNameSpaces) ? " (a language module)" : "");
    } else if (c.equals(Object.class)) {
      return "unknown";
    } else if (c.equals(String.class)) {
      return "string";
    } else if (c.equals(Integer.class)) {
      return "int";
    } else if (c.equals(Boolean.class)) {
      return "bool";
    } else if (List.class.isAssignableFrom(c)) { // This is a Java List that isn't a SkylarkList
      return "List"; // This case shouldn't happen in normal code, but we keep it for debugging.
    } else if (Map.class.isAssignableFrom(c)) { // This is a Java Map that isn't a SkylarkDict
      return "Map"; // This case shouldn't happen in normal code, but we keep it for debugging.
    } else if (StarlarkFunction.class.isAssignableFrom(c)) {
      return "function";
    } else if (c.equals(SelectorValue.class)) {
      return "select";
    } else if (NestedSet.class.isAssignableFrom(c)) {
      // TODO(bazel-team): no one should be seeing naked NestedSet at all.
      return "depset";
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
      throw new EvalException(
          expr.getLocation(),
          "unexpected null value, please send a bug report. "
              + "This was generated by expression '"
              + expr
              + "'");
    }
    return obj;
  }

  /**
   * Returns the truth value of an object, according to Python rules.
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
      return !Iterables.isEmpty((Iterable<?>) o);
    } else {
      return true;
    }
  }

  public static Collection<?> toCollection(Object o, Location loc, @Nullable Environment env)
      throws EvalException {
    if (o instanceof Collection) {
      return (Collection<?>) o;
    } else if (o instanceof SkylarkList) {
      return ((SkylarkList) o).getImmutableList();
    } else if (o instanceof Map) {
      // For dictionaries we iterate through the keys only
      if (o instanceof SkylarkDict) {
        // SkylarkDicts handle ordering themselves
        SkylarkDict<?, ?> dict = (SkylarkDict) o;
        List<Object> list = Lists.newArrayListWithCapacity(dict.size());
        for (Map.Entry<?, ?> entries : dict.entrySet()) {
          list.add(entries.getKey());
        }
        return ImmutableList.copyOf(list);
      }
      // For determinism, we sort the keys.
      try {
        return SKYLARK_COMPARATOR.sortedCopy(((Map<?, ?>) o).keySet());
      } catch (ComparisonException e) {
        throw new EvalException(loc, e);
      }
    } else if (o instanceof SkylarkNestedSet) {
      return nestedSetToCollection((SkylarkNestedSet) o, loc, env);
    } else {
      throw new EvalException(loc,
          "type '" + getDataTypeName(o) + "' is not a collection");
    }
  }

  private static Collection<?> nestedSetToCollection(
      SkylarkNestedSet set, Location loc, @Nullable Environment env) throws EvalException {
    if (env != null && env.getSemantics().incompatibleDepsetIsNotIterable()) {
      throw new EvalException(
          loc,
          "type 'depset' is not iterable. Use the `to_list()` method to get a list. Use "
              + "--incompatible_depset_is_not_iterable=false to temporarily disable this check.");
    }
    return set.toCollection();
  }

  public static Iterable<?> toIterable(Object o, Location loc, @Nullable Environment env)
      throws EvalException {
    if (o instanceof String) {
      // This is not as efficient as special casing String in for and dict and list comprehension
      // statements. However this is a more unified way.
      return split((String) o, loc, env);
    } else if (o instanceof SkylarkNestedSet) {
      return nestedSetToCollection((SkylarkNestedSet) o, loc, env);
    } else if (o instanceof Iterable) {
      return (Iterable<?>) o;
    } else if (o instanceof Map) {
      return toCollection(o, loc, env);
    } else {
      throw new EvalException(loc,
          "type '" + getDataTypeName(o) + "' is not iterable");
    }
  }

  /**
   * Given an {@link Iterable}, returns it as-is. Given a {@link SkylarkNestedSet}, returns its
   * contents as an iterable. Throws {@link EvalException} for any other value.
   *
   * <p>This is a kludge for the change that made {@code SkylarkNestedSet} not implement {@code
   * Iterable}. It is different from {@link #toIterable} in its behavior for strings and other types
   * that are not strictly Java-iterable.
   *
   * @throws EvalException if {@code o} is not an iterable or set
   * @deprecated avoid writing APIs that implicitly treat depsets as iterables. It encourages
   *     unnecessary flattening of depsets.
   *     <p>TODO(bazel-team): Remove this if/when implicit iteration over {@code SkylarkNestedSet}
   *     is no longer supported.
   */
  @Deprecated
  public static Iterable<?> toIterableStrict(Object o, Location loc, @Nullable Environment env)
      throws EvalException {
    if (o instanceof Iterable) {
      return (Iterable<?>) o;
    } else if (o instanceof SkylarkNestedSet) {
      return nestedSetToCollection((SkylarkNestedSet) o, loc, env);
    } else {
      throw new EvalException(loc,
          "expected Iterable or depset, but got '" + getDataTypeName(o) + "' (strings and maps "
          + "are not allowed here)");
    }
  }

  public static void lock(Object object, Location loc) {
    if (object instanceof SkylarkMutable) {
      ((SkylarkMutable) object).lock(loc);
    }
  }

  public static void unlock(Object object, Location loc) {
    if (object instanceof SkylarkMutable) {
      ((SkylarkMutable) object).unlock(loc);
    }
  }

  private static ImmutableList<String> split(String value, Location loc, @Nullable Environment env)
      throws EvalException {
    if (env != null && env.getSemantics().incompatibleStringIsNotIterable()) {
      throw new EvalException(
          loc,
          "type 'string' is not iterable. You may still use `len` and string indexing. Use "
              + "--incompatible_string_is_not_iterable=false to temporarily disable this check.");
    }

    ImmutableList.Builder<String> builder = ImmutableList.builderWithExpectedSize(value.length());
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
      return ((SkylarkList<?>) arg).size();
    } else if (arg instanceof SkylarkNestedSet) {
      // TODO(bazel-team): Add a deprecation warning: don't implicitly flatten depsets.
      return ((SkylarkNestedSet) arg).toCollection().size();
    } else if (arg instanceof Iterable) {
      // Iterables.size() checks if arg is a Collection so it's efficient in that sense.
      return Iterables.size((Iterable<?>) arg);
    }
    return -1;
  }

  // The following functions for indexing and slicing match the behavior of Python.

  /**
   * Resolves a positive or negative index to an index in the range [0, length), or throws
   * EvalException if it is out-of-range. If the index is negative, it counts backward from
   * length.
   */
  public static int getSequenceIndex(int index, int length, Location loc)
      throws EvalException {
    int actualIndex = index;
    if (actualIndex < 0) {
      actualIndex += length;
    }
    if (actualIndex < 0 || actualIndex >= length) {
      throw new EvalException(
          loc,
          "index out of range (index is " + index + ", but sequence has " + length + " elements)");
    }
    return actualIndex;
  }

  /**
   * Performs index resolution after verifying that the given object has index type.
   */
  public static int getSequenceIndex(Object index, int length, Location loc)
      throws EvalException {
    if (!(index instanceof Integer)) {
      throw new EvalException(
          loc, "indices must be integers, not " + EvalUtils.getDataTypeName(index));
    }
    return getSequenceIndex(((Integer) index).intValue(), length, loc);
  }

  /**
   * Resolves a positive or negative index to an integer that can denote the left or right boundary
   * of a slice. If reverse is false, the slice has positive stride (i.e., its elements are in their
   * normal order) and the result is guaranteed to be in range [0, length + 1). If reverse is true,
   * the slice has negative stride and the result is in range [-1, length). In either case, if the
   * index is negative, it counts backward from length. Note that an input index of -1 represents
   * the last element's position, while an output integer of -1 represents the imaginary position
   * to the left of the first element.
   */
  public static int clampRangeEndpoint(int index, int length, boolean reverse) {
    if (index < 0) {
      index += length;
    }
    if (!reverse) {
      return Math.max(Math.min(index, length), 0);
    } else {
      return Math.max(Math.min(index, length - 1), -1);
    }
  }

  /**
   * Resolves a positive or negative index to an integer that can denote the boundary for a
   * slice with positive stride.
   */
  public static int clampRangeEndpoint(int index, int length) {
    return clampRangeEndpoint(index, length, false);
  }

  /**
   * Calculates the indices of the elements that should be included in the slice [start:end:step]
   * of a sequence with the given length. Each of start, end, and step must be supplied, and step
   * may not be 0.
   */
  public static List<Integer> getSliceIndices(int start, int end, int step, int length) {
    if (step == 0) {
      throw new IllegalArgumentException("Slice step cannot be zero");
    }
    start = clampRangeEndpoint(start, length, step < 0);
    end = clampRangeEndpoint(end, length, step < 0);
    // precise computation is slightly more involved, but since it can overshoot only by a single
    // element it's fine
    final int expectedMaxSize = Math.abs(start - end) / Math.abs(step) + 1;
    ImmutableList.Builder<Integer> indices = ImmutableList.builderWithExpectedSize(expectedMaxSize);
    for (int current = start; step > 0 ? current < end : current > end; current += step) {
      indices.add(current);
    }
    return indices.build();
  }

  /**
   * Calculates the indices of the elements in a slice, after validating the arguments and replacing
   * Runtime.NONE with default values. Throws an EvalException if a bad argument is given.
   */
  public static List<Integer> getSliceIndices(
      Object startObj, Object endObj, Object stepObj, int length, Location loc)
      throws EvalException {
    int start;
    int end;
    int step;

    if (stepObj == Runtime.NONE) {
      step = 1;
    } else if (stepObj instanceof Integer) {
      step = ((Integer) stepObj).intValue();
    } else {
      throw new EvalException(
          loc, String.format("slice step must be an integer, not '%s'", stepObj));
    }
    if (step == 0) {
      throw new EvalException(loc, "slice step cannot be zero");
    }

    if (startObj == Runtime.NONE) {
      start = (step > 0) ? 0 : length - 1;
    } else if (startObj instanceof Integer) {
      start = ((Integer) startObj).intValue();
    } else {
      throw new EvalException(
          loc, String.format("slice start must be an integer, not '%s'", startObj));
    }
    if (endObj == Runtime.NONE) {
      // If step is negative, can't use -1 for end since that would be converted
      // to the rightmost element's position.
      end = (step > 0) ? length : -length - 1;
    } else if (endObj instanceof Integer) {
      end = ((Integer) endObj).intValue();
    } else {
      throw new EvalException(loc, String.format("slice end must be an integer, not '%s'", endObj));
    }

    return getSliceIndices(start, end, step, length);
  }

  /** @return true if x is Java null or Skylark None */
  public static boolean isNullOrNone(Object x) {
    return x == null || x == Runtime.NONE;
  }

  /**
   * Build a SkylarkDict of kwarg arguments from a list, removing null-s or None-s.
   *
   * @param env the Environment in which this map can be mutated.
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
  public static <K, V> SkylarkDict<K, V> optionMap(Environment env, Object... init) {
    ImmutableMap.Builder<K, V> b = new ImmutableMap.Builder<>();
    Preconditions.checkState(init.length % 2 == 0);
    for (int i = init.length - 2; i >= 0; i -= 2) {
      K key = (K) Preconditions.checkNotNull(init[i]);
      V value = (V) init[i + 1];
      if (!isNullOrNone(value)) {
        b.put(key, value);
      }
    }
    return SkylarkDict.copyOf(env, b.build());
  }
}
