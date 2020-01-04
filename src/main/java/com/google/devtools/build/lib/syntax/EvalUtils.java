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
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkInterfaceUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.Concatable.Concatter;
import com.google.devtools.build.lib.util.SpellChecker;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Utilities used by the evaluator. */
// TODO(adonovan): rename this class to Starlark. Its API should contain all the fundamental values
// and operators of the language: None, len, truth, str, iterate, equal, compare, getattr, index,
// slice, parse, exec, eval, and so on.
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
        private int compareLists(Sequence<?> o1, Sequence<?> o2) {
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

          // optimize the most common cases

          if (o1 instanceof String && o2 instanceof String) {
            return ((String) o1).compareTo((String) o2);
          }
          if (o1 instanceof Integer && o2 instanceof Integer) {
            return Integer.compare((Integer) o1, (Integer) o2);
          }

          o1 = Starlark.fromJava(o1, null);
          o2 = Starlark.fromJava(o2, null);

          if (o1 instanceof Sequence
              && o2 instanceof Sequence
              && o1 instanceof Tuple == o2 instanceof Tuple) {
            return compareLists((Sequence) o1, (Sequence) o2);
          }

          if (o1 instanceof ClassObject) {
            throw new ComparisonException("Cannot compare structs");
          }
          if (o1 instanceof Depset) {
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

  /** Throws EvalException if x is not hashable. */
  static void checkHashable(Object x) throws EvalException {
    if (!isHashable(x)) {
      // This results in confusing errors such as "unhashable type: tuple".
      // TODO(adonovan): ideally the error message would explain which
      // element of, say, a tuple is unhashable. The only practical way
      // to implement this is by implementing isHashable as a call to
      // Object.hashCode within a try/catch, and requiring all
      // unhashable Starlark values to throw a particular unchecked exception
      // with a helpful error message.
      throw Starlark.errorf("unhashable type: '%s'", EvalUtils.getDataTypeName(x));
    }
  }

  /**
   * Is this object known or assumed to be recursively hashable by Skylark?
   *
   * @param o an Object
   * @return true if the object is known to be a hashable value.
   */
  public static boolean isHashable(Object o) {
    if (o instanceof StarlarkValue) {
      return ((StarlarkValue) o).isHashable();
    }
    return isImmutable(o.getClass());
  }

  /**
   * Is this object known or assumed to be recursively immutable by Skylark?
   *
   * @param o an Object
   * @return true if the object is known to be an immutable value.
   */
  // NB: This is used as the basis for accepting objects in Depset-s.
  public static boolean isImmutable(Object o) {
    if (o instanceof StarlarkValue) {
      return ((StarlarkValue) o).isImmutable();
    }
    return isImmutable(o.getClass());
  }

  /**
   * Is this class known to be *recursively* immutable by Skylark? For instance, class Tuple is not
   * it, because it can contain mutable values.
   *
   * @param c a Class
   * @return true if the class is known to represent only recursively immutable values.
   */
  // NB: This is used as the basis for accepting objects in Depset-s,
  // as well as for accepting objects as keys for Skylark dict-s.
  private static boolean isImmutable(Class<?> c) {
    return c.isAnnotationPresent(Immutable.class) // TODO(bazel-team): beware of containers!
        || c.equals(String.class)
        || c.equals(Integer.class)
        || c.equals(Boolean.class);
  }

  // TODO(bazel-team): move the following few type-related functions to SkylarkType
  /**
   * Return the Skylark-type of {@code c}
   *
   * <p>The result will be a type that Skylark understands and is either equal to {@code c} or is a
   * supertype of it.
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
    // (String/Boolean/Integer) to implement StarlarkValue. We should also require them to have a
    // (possibly inherited) @SkylarkModule annotation.
    Class<?> parent = SkylarkInterfaceUtils.getParentWithSkylarkModule(c);
    if (parent != null) {
      return parent;
    }
    Preconditions.checkArgument(
        StarlarkValue.class.isAssignableFrom(c),
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
      if (object instanceof Depset) {
        Depset set = (Depset) object;
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
    } else if (List.class.isAssignableFrom(c)) { // This is a Java List that isn't a Sequence
      return "List"; // This case shouldn't happen in normal code, but we keep it for debugging.
    } else if (Map.class.isAssignableFrom(c)) { // This is a Java Map that isn't a Dict
      return "Map"; // This case shouldn't happen in normal code, but we keep it for debugging.
    } else if (StarlarkCallable.class.isAssignableFrom(c)) {
      // TODO(adonovan): each StarlarkCallable should report its own type string.
      return "function";
    } else if (c.equals(SelectorValue.class)) {
      return "select";
    } else {
      if (c.getSimpleName().isEmpty()) {
        return c.getName();
      } else {
        return c.getSimpleName();
      }
    }
  }

  public static void lock(Object object, Location loc) {
    if (object instanceof StarlarkMutable) {
      StarlarkMutable x = (StarlarkMutable) object;
      x.mutability().lock(x, loc);
    }
  }

  public static void unlock(Object object, Location loc) {
    if (object instanceof StarlarkMutable) {
      StarlarkMutable x = (StarlarkMutable) object;
      x.mutability().unlock(x, loc);
    }
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
      throw new EvalException(loc, "indices must be integers, not " + getDataTypeName(index));
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
   * Starlark.NONE with default values. Throws an EvalException if a bad argument is given.
   */
  public static List<Integer> getSliceIndices(
      Object startObj, Object endObj, Object stepObj, int length, Location loc)
      throws EvalException {
    int start;
    int end;
    int step;

    if (stepObj == Starlark.NONE) {
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

    if (startObj == Starlark.NONE) {
      start = (step > 0) ? 0 : length - 1;
    } else if (startObj instanceof Integer) {
      start = ((Integer) startObj).intValue();
    } else {
      throw new EvalException(
          loc, String.format("slice start must be an integer, not '%s'", startObj));
    }
    if (endObj == Starlark.NONE) {
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
    return x == null || x == Starlark.NONE;
  }

  /**
   * Installs a global hook that causes subsequently executed Starlark threads to notify the
   * debugger of important events. Closes any previously set debugger. Call {@code
   * setDebugger(null)} to disable debugging.
   */
  public static void setDebugger(Debugger dbg) {
    Eval.setDebugger(dbg);
  }

  /** Returns the named field or method of value {@code x}, or null if not found. */
  static Object getAttr(StarlarkThread thread, Location loc, Object x, String name)
      throws EvalException, InterruptedException {
    // @SkylarkCallable-annotated field or method?
    MethodDescriptor method = CallUtils.getMethod(thread.getSemantics(), x.getClass(), name);
    if (method != null) {
      if (method.isStructField()) {
        return method.callField(x, loc, thread.getSemantics(), thread.mutability());
      } else {
        return new BuiltinCallable(x, name, method);
      }
    }

    // user-defined field?
    if (x instanceof ClassObject) {
      // TODO(adonovan): get rid of SkylarkClassObject once
      // --incompatible_disable_target_provider_fields goes away,
      // then remove the thread and loc parameters and publish
      // this method as Starlark.getattr.
      Object field =
          x instanceof SkylarkClassObject
              ? ((SkylarkClassObject) x).getValue(loc, thread.getSemantics(), name)
              : ((ClassObject) x).getValue(name);
      if (field != null) {
        return Starlark.checkValid(field);
      }
    }

    return null;
  }

  static EvalException getMissingAttrException(
      Object object, String name, StarlarkSemantics semantics) {
    String suffix = "";
    if (object instanceof ClassObject) {
      String customErrorMessage = ((ClassObject) object).getErrorMessageForUnknownField(name);
      if (customErrorMessage != null) {
        return Starlark.errorf("%s", customErrorMessage);
      }
      suffix = SpellChecker.didYouMean(name, ((ClassObject) object).getFieldNames());
    } else {
      suffix = SpellChecker.didYouMean(name, CallUtils.getFieldNames(semantics, object));
    }
    return Starlark.errorf(
        "'%s' value has no field or method '%s'%s", getDataTypeName(object), name, suffix);
  }

  /** Evaluates an eager binary operation, {@code x op y}. (Excludes AND and OR.) */
  static Object binaryOp(TokenKind op, Object x, Object y, StarlarkThread thread, Location location)
      throws EvalException, InterruptedException {
    try {
      switch (op) {
        case PLUS:
          return plus(x, y, thread, location);

        case PIPE:
          return pipe(x, y, thread, location);

        case AMPERSAND:
          return and(x, y, location);

        case CARET:
          return xor(x, y, location);

        case GREATER_GREATER:
          return rightShift(x, y, location);

        case LESS_LESS:
          return leftShift(x, y, location);

        case MINUS:
          return minus(x, y, location);

        case STAR:
          return mult(x, y, thread, location);

        case SLASH:
          throw new EvalException(
              location,
              "The `/` operator is not allowed. Please use the `//` operator for integer "
                  + "division.");

        case SLASH_SLASH:
          return divide(x, y, location);

        case PERCENT:
          return percent(x, y, location);

        case EQUALS_EQUALS:
          return x.equals(y);

        case NOT_EQUALS:
          return !x.equals(y);

        case LESS:
          return compare(x, y, location) < 0;

        case LESS_EQUALS:
          return compare(x, y, location) <= 0;

        case GREATER:
          return compare(x, y, location) > 0;

        case GREATER_EQUALS:
          return compare(x, y, location) >= 0;

        case IN:
          return in(x, y, thread, location);

        case NOT_IN:
          return !in(x, y, thread, location);

        default:
          throw new AssertionError("Unsupported binary operator: " + op);
      }
    } catch (ArithmeticException e) {
      throw new EvalException(location, e.getMessage());
    }
  }

  /** Implements comparison operators. */
  private static int compare(Object x, Object y, Location location) throws EvalException {
    try {
      return SKYLARK_COMPARATOR.compare(x, y);
    } catch (ComparisonException e) {
      throw new EvalException(location, e);
    }
  }

  /** Implements 'x + y'. */
  static Object plus(Object x, Object y, StarlarkThread thread, Location location)
      throws EvalException {
    // int + int
    if (x instanceof Integer && y instanceof Integer) {
      return Math.addExact((Integer) x, (Integer) y);
    }

    // string + string
    if (x instanceof String && y instanceof String) {
      return (String) x + (String) y;
    }

    if (x instanceof SelectorValue
        || y instanceof SelectorValue
        || x instanceof SelectorList
        || y instanceof SelectorList) {
      return SelectorList.concat(location, x, y);
    }

    if (x instanceof Tuple && y instanceof Tuple) {
      return Tuple.concat((Tuple<?>) x, (Tuple<?>) y);
    }

    if (x instanceof StarlarkList && y instanceof StarlarkList) {
      return StarlarkList.concat((StarlarkList<?>) x, (StarlarkList<?>) y, thread.mutability());
    }

    if (x instanceof Concatable && y instanceof Concatable) {
      Concatable lobj = (Concatable) x;
      Concatable robj = (Concatable) y;
      Concatter concatter = lobj.getConcatter();
      if (concatter != null && concatter.equals(robj.getConcatter())) {
        return concatter.concat(lobj, robj, location);
      } else {
        throw unknownBinaryOperator(x, y, TokenKind.PLUS, location);
      }
    }

    // TODO(bazel-team): Remove deprecated operator.
    if (x instanceof Depset) {
      if (thread.getSemantics().incompatibleDepsetUnion()) {
        throw new EvalException(
            location,
            "`+` operator on a depset is forbidden. See "
                + "https://docs.bazel.build/versions/master/skylark/depsets.html for "
                + "recommendations. Use --incompatible_depset_union=false "
                + "to temporarily disable this check.");
      }
      return Depset.unionOf((Depset) x, y);
    }
    throw unknownBinaryOperator(x, y, TokenKind.PLUS, location);
  }

  /** Implements 'x | y'. */
  private static Object pipe(Object x, Object y, StarlarkThread thread, Location location)
      throws EvalException {
    if (x instanceof Integer && y instanceof Integer) {
      return ((Integer) x) | ((Integer) y);
    } else if (x instanceof Depset) {
      if (thread.getSemantics().incompatibleDepsetUnion()) {
        throw new EvalException(
            location,
            "`|` operator on a depset is forbidden. See "
                + "https://docs.bazel.build/versions/master/skylark/depsets.html for "
                + "recommendations. Use --incompatible_depset_union=false "
                + "to temporarily disable this check.");
      }
      return Depset.unionOf((Depset) x, y);
    }
    throw unknownBinaryOperator(x, y, TokenKind.PIPE, location);
  }

  /** Implements 'x - y'. */
  private static Object minus(Object x, Object y, Location location) throws EvalException {
    if (x instanceof Integer && y instanceof Integer) {
      return Math.subtractExact((Integer) x, (Integer) y);
    }
    throw unknownBinaryOperator(x, y, TokenKind.MINUS, location);
  }

  /** Implements 'x * y'. */
  private static Object mult(Object x, Object y, StarlarkThread thread, Location location)
      throws EvalException {
    Integer number = null;
    Object otherFactor = null;

    if (x instanceof Integer) {
      number = (Integer) x;
      otherFactor = y;
    } else if (y instanceof Integer) {
      number = (Integer) y;
      otherFactor = x;
    }

    if (number != null) {
      if (otherFactor instanceof Integer) {
        return Math.multiplyExact(number, (Integer) otherFactor);
      } else if (otherFactor instanceof String) {
        return Strings.repeat((String) otherFactor, Math.max(0, number));
      } else if (otherFactor instanceof Tuple) {
        return ((Tuple<?>) otherFactor).repeat(number, thread.mutability());
      } else if (otherFactor instanceof StarlarkList) {
        return ((StarlarkList<?>) otherFactor).repeat(number, thread.mutability());
      }
    }
    throw unknownBinaryOperator(x, y, TokenKind.STAR, location);
  }

  /** Implements 'x // y'. */
  private static Object divide(Object x, Object y, Location location) throws EvalException {
    // int / int
    if (x instanceof Integer && y instanceof Integer) {
      if (y.equals(0)) {
        throw new EvalException(location, "integer division by zero");
      }
      // Integer division doesn't give the same result in Java and in Python 2 with
      // negative numbers.
      // Java:   -7/3 = -2
      // Python: -7/3 = -3
      // We want to follow Python semantics, so we use float division and round down.
      return (int) Math.floor(Double.valueOf((Integer) x) / (Integer) y);
    }
    throw unknownBinaryOperator(x, y, TokenKind.SLASH_SLASH, location);
  }

  /** Implements 'x % y'. */
  private static Object percent(Object x, Object y, Location location) throws EvalException {
    // int % int
    if (x instanceof Integer && y instanceof Integer) {
      if (y.equals(0)) {
        throw new EvalException(location, "integer modulo by zero");
      }
      // Python and Java implement division differently, wrt negative numbers.
      // In Python, sign of the result is the sign of the divisor.
      int div = (Integer) y;
      int result = ((Integer) x).intValue() % Math.abs(div);
      if (result > 0 && div < 0) {
        result += div; // make the result negative
      } else if (result < 0 && div > 0) {
        result += div; // make the result positive
      }
      return result;
    }

    // string % tuple, string % dict, string % anything-else
    if (x instanceof String) {
      String pattern = (String) x;
      try {
        if (y instanceof Tuple) {
          return Starlark.formatWithList(pattern, (Tuple) y);
        }
        return Starlark.format(pattern, y);
      } catch (IllegalFormatException e) {
        throw new EvalException(location, e.getMessage());
      }
    }
    throw unknownBinaryOperator(x, y, TokenKind.PERCENT, location);
  }

  /** Implements 'x & y'. */
  private static Object and(Object x, Object y, Location location) throws EvalException {
    if (x instanceof Integer && y instanceof Integer) {
      return (Integer) x & (Integer) y;
    }
    throw unknownBinaryOperator(x, y, TokenKind.AMPERSAND, location);
  }

  /** Implements 'x ^ y'. */
  private static Object xor(Object x, Object y, Location location) throws EvalException {
    if (x instanceof Integer && y instanceof Integer) {
      return (Integer) x ^ (Integer) y;
    }
    throw unknownBinaryOperator(x, y, TokenKind.CARET, location);
  }

  /** Implements 'x >> y'. */
  private static Object rightShift(Object x, Object y, Location location) throws EvalException {
    if (x instanceof Integer && y instanceof Integer) {
      if ((Integer) y < 0) {
        throw new EvalException(location, "negative shift count: " + y);
      } else if ((Integer) y >= Integer.SIZE) {
        return ((Integer) x < 0) ? -1 : 0;
      }
      return (Integer) x >> (Integer) y;
    }
    throw unknownBinaryOperator(x, y, TokenKind.GREATER_GREATER, location);
  }

  /** Implements 'x << y'. */
  private static Object leftShift(Object x, Object y, Location location) throws EvalException {
    if (x instanceof Integer && y instanceof Integer) {
      if ((Integer) y < 0) {
        throw new EvalException(location, "negative shift count: " + y);
      }
      Integer result = (Integer) x << (Integer) y;
      if (!rightShift(result, y, location).equals(x)) {
        throw new ArithmeticException("integer overflow");
      }
      return result;
    }
    throw unknownBinaryOperator(x, y, TokenKind.LESS_LESS, location);
  }

  /** Implements 'x in y'. */
  private static boolean in(Object x, Object y, StarlarkThread thread, Location location)
      throws EvalException {
    if (y instanceof SkylarkQueryable) {
      return ((SkylarkQueryable) y).containsKey(x, location, thread);
    } else if (y instanceof String) {
      if (x instanceof String) {
        return ((String) y).contains((String) x);
      } else {
        throw new EvalException(
            location,
            "'in <string>' requires string as left operand, not '" + getDataTypeName(x) + "'");
      }
    } else {
      throw new EvalException(
          location,
          "argument of type '"
              + getDataTypeName(y)
              + "' is not iterable. "
              + "in operator only works on lists, tuples, dicts and strings.");
    }
  }

  /** Returns an exception signifying incorrect types for the given operator. */
  private static EvalException unknownBinaryOperator(
      Object x, Object y, TokenKind op, Location location) {
    // NB: this message format is identical to that used by CPython 2.7.6 or 3.4.0,
    // though python raises a TypeError.
    // TODO(adonovan): make error more concise: "unsupported binary op: list + int".
    return new EvalException(
        location,
        String.format(
            "unsupported operand type(s) for %s: '%s' and '%s'",
            op, getDataTypeName(x), getDataTypeName(y)));
  }

  /** Evaluates a unary operation. */
  static Object unaryOp(TokenKind op, Object x, Location loc)
      throws EvalException, InterruptedException {
    switch (op) {
      case NOT:
        return !Starlark.truth(x);

      case MINUS:
        if (x instanceof Integer) {
          try {
            return Math.negateExact((Integer) x);
          } catch (ArithmeticException e) {
            // Fails for -MIN_INT.
            throw new EvalException(loc, e.getMessage());
          }
        }
        break;

      case PLUS:
        if (x instanceof Integer) {
          return x;
        }
        break;

      case TILDE:
        if (x instanceof Integer) {
          return ~((Integer) x);
        }
        break;

      default:
        /* fall through */
    }
    throw new EvalException(
        loc, String.format("unsupported unary operation: %s%s", op, getDataTypeName(x)));
  }

  /**
   * Returns the element of sequence or mapping {@code object} indexed by {@code key}.
   *
   * @throws EvalException if {@code object} is not a sequence or mapping.
   */
  public static Object index(Object object, Object key, StarlarkThread thread, Location loc)
      throws EvalException, InterruptedException {
    if (object instanceof SkylarkIndexable) {
      Object result = ((SkylarkIndexable) object).getIndex(key, loc);
      // TODO(bazel-team): We shouldn't have this fromJava call here. If it's needed at all,
      // it should go in the implementations of SkylarkIndexable#getIndex that produce non-Skylark
      // values.
      return result == null ? null : Starlark.fromJava(result, thread.mutability());
    } else if (object instanceof String) {
      String string = (String) object;
      int index = getSequenceIndex(key, string.length(), loc);
      return string.substring(index, index + 1);
    } else {
      throw new EvalException(
          loc,
          String.format(
              "type '%s' has no operator [](%s)", getDataTypeName(object), getDataTypeName(key)));
    }
  }

  /**
   * Parses the input as a file, validates it in the {@code thread.getGlobals} environment using
   * options defined by {@code thread.getSemantics}, and returns the syntax tree. It uses Starlark
   * (not BUILD) validation semantics.
   *
   * <p>The thread is primarily used for its GlobalFrame. Scan/parse/validate errors are recorded in
   * the StarlarkFile. It is the caller's responsibility to inspect them.
   */
  public static StarlarkFile parseAndValidateSkylark(ParserInput input, StarlarkThread thread) {
    StarlarkFile file = StarlarkFile.parse(input);
    ValidationEnvironment.validateFile(
        file, thread.getGlobals(), thread.getSemantics(), /*isBuildFile=*/ false);
    return file;
  }

  /**
   * Parses the input as a file, validates it in the {@code thread.getGlobals} environment using
   * options defined by {@code thread.getSemantics}, and executes it. It uses Starlark (not BUILD)
   * validation semantics.
   */
  public static void exec(ParserInput input, StarlarkThread thread)
      throws SyntaxError, EvalException, InterruptedException {
    StarlarkFile file = parseAndValidateSkylark(input, thread);
    if (!file.ok()) {
      throw new SyntaxError(file.errors());
    }
    // TODO(adonovan): turn toplevel statements into a StarlarkFunction, and call it.
    // This ensures we have an entry in the call stack even for the toplevel.
    exec(file, thread);
  }

  /** Executes a parsed, validated Starlark file in a given StarlarkThread. */
  public static void exec(StarlarkFile file, StarlarkThread thread)
      throws EvalException, InterruptedException {
    Eval.execFile(thread, file);
  }

  /**
   * Parses the input as an expression, validates it in the {@code thread.getGlobals} environment
   * using options defined by {@code thread.getSemantics}, and evaluates it. It uses Starlark (not
   * BUILD) validation semantics.
   */
  public static Object eval(ParserInput input, StarlarkThread thread)
      throws SyntaxError, EvalException, InterruptedException {
    Expression expr = Expression.parse(input);
    ValidationEnvironment.validateExpr(expr, thread.getGlobals(), thread.getSemantics());
    // TODO(adonovan): turn expr into a StarlarkFunction, and call it.
    // This ensures we have an entry in the call stack even for the toplevel.
    return Eval.eval(thread, expr);
  }

  /**
   * Parses the input as a file, validates it in the {@code thread.getGlobals} environment using
   * options defined by {@code thread.getSemantics}, and executes it. The function uses Starlark
   * (not BUILD) validation semantics. If the final statement is an expression statement, it returns
   * the value of that expression, otherwise it returns null.
   *
   * <p>The function's name is intentionally unattractive. Don't call it unless you're accepting
   * strings from an interactive user interface such as a REPL or debugger; use {@link #exec} or
   * {@link #eval} instead.
   */
  @Nullable
  public static Object execAndEvalOptionalFinalExpression(ParserInput input, StarlarkThread thread)
      throws SyntaxError, EvalException, InterruptedException {
    StarlarkFile file = StarlarkFile.parse(input);
    List<Statement> stmts = file.getStatements();
    Expression expr = null;
    int n = stmts.size();
    if (n > 0 && stmts.get(n - 1) instanceof ExpressionStatement) {
      expr = ((ExpressionStatement) stmts.get(n - 1)).getExpression();
      stmts = stmts.subList(0, n - 1);
    }
    ValidationEnvironment.validateFile(
        file, thread.getGlobals(), thread.getSemantics(), /*isBuildFile=*/ false);
    if (!file.ok()) {
      throw new SyntaxError(file.errors());
    }
    Eval.execStatements(thread, stmts);
    return expr != null ? Eval.eval(thread, expr) : null;
  }
}
